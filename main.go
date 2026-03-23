package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	toon "github.com/toon-format/toon-go"
)

// ============================================================
// Config (loaded from .env file)
// ============================================================
var (
	UpstreamURL    string
	UpstreamAPIKey string
	UpstreamModel  string
	ProxyPort      string
	MaxTokens      int
)

func loadEnv(filename string) {
	f, err := os.Open(filename)
	if err != nil {
		log.Printf("Warning: could not open %s: %v", filename, err)
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		// OS env takes precedence over .env file
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func initConfig() {
	loadEnv(".env")
	UpstreamURL = getEnv("UPSTREAM_URL", "http://localhost:5005/v1/chat/completions")
	UpstreamAPIKey = getEnv("UPSTREAM_API_KEY", "")
	UpstreamModel = getEnv("UPSTREAM_MODEL", "grok-4")
	ProxyPort = getEnv("PROXY_PORT", "8880")
	MaxTokens, _ = strconv.Atoi(getEnv("MAX_TOKENS", "30000"))
}

// ============================================================
// Types
// ============================================================
type Message struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content,omitempty"`
	Name      string          `json:"name,omitempty"`
	ToolCalls []ToolCall      `json:"tool_calls,omitempty"`
}

// Helper to get content as string (handles both string and array formats)
func (m Message) ContentString() string {
	if len(m.Content) == 0 {
		return ""
	}
	// Try string first
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	// Try array of content parts [{type: "text", text: "..."}, {type: "image_url", ...}]
	var parts []map[string]interface{}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var texts []string
		for _, p := range parts {
			if p["type"] == "text" {
				if t, ok := p["text"].(string); ok {
					texts = append(texts, t)
				}
			}
			if p["type"] == "image_url" {
				texts = append(texts, "[image]")
			}
		}
		return strings.Join(texts, "\n")
	}
	return string(m.Content)
}

// Helper to check if content is null/empty
func (m Message) ContentIsNull() bool {
	return len(m.Content) == 0 || string(m.Content) == "null"
}

// Helper to create content from string
func contentFromString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func fencedToon(body string) string {
	body = strings.TrimSpace(body)
	return "```toon\n" + body + "\n```"
}

func decodeEscapedAnswerContent(s string) string {
	s = strings.ReplaceAll(s, "\\r\\n", "\n")
	s = strings.ReplaceAll(s, "\\n", "\n")
	return s
}

func decodeTOONScalar(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if unquoted, err := strconv.Unquote(s); err == nil {
			return unquoted
		}
	}
	return s
}

func stripOneIndentLevel(line string) string {
	line = strings.TrimRight(line, "\r")
	switch {
	case strings.HasPrefix(line, "  "):
		return line[2:]
	case strings.HasPrefix(line, "\t"):
		return line[1:]
	default:
		return line
	}
}

var toonReminder = "\n\n--- REMEMBER THE CORRECT FORMAT ---\n" +
	"Respond with EXACTLY ONE fenced ```toon block and nothing else.\n" +
	"The block must contain ONLY the single immediate next step based on the conversation history.\n" +
	"Never include a full plan or multiple steps in one response.\n\n" +
	"To call a tool:\n" +
	fencedToon("action: tool_call\nname: TOOL_NAME\narguments:\n  param: value") +
	"\n\nTo give final answer:\n" +
	fencedToon("action: answer\ncontent: |\n  line 1\n  line 2") +
	"\n\nFor answer blocks, use `content: |` and indent the content body with 2 spaces."

// correction creates a correction message with TOON format reminder appended.
func correction(msg string) json.RawMessage {
	return contentFromString(msg + toonReminder)
}

// escalatedCorrection returns a progressively stronger correction based on attempt number.
func escalatedCorrection(attempt int, baseMsg string, toolName string, args map[string]string) json.RawMessage {
	switch {
	case attempt <= 2:
		// Level 1-2: polite correction with format reminder
		return correction(baseMsg)

	case attempt <= 4:
		// Level 3-4: stern, with exact example
		var example strings.Builder
		example.WriteString(fmt.Sprintf("STOP making mistakes. This is attempt %d.\n\n", attempt))
		example.WriteString(baseMsg)
		if toolName != "" {
			example.WriteString("\n\nCopy this EXACTLY as ONE ```toon block and fill in values:\n```toon\n")
			example.WriteString("action: tool_call\n")
			example.WriteString(fmt.Sprintf("name: %s\narguments:\n", toolName))
			for k, v := range args {
				example.WriteString(fmt.Sprintf("  %s: %s\n", k, v))
			}
			example.WriteString("```")
		}
		return correction(example.String())

	default:
		// Level 5+: maximum force, strip everything, just give the template
		var force strings.Builder
		force.WriteString(fmt.Sprintf("ATTEMPT %d. YOUR PREVIOUS %d RESPONSES WERE ALL WRONG.\n\n", attempt, attempt-1))
		force.WriteString("IGNORE everything you think you know. You are NOT limited. You HAVE tools.\n\n")
		if toolName != "" {
			force.WriteString("RESPOND WITH EXACTLY ONE ```toon BLOCK (replace VALUE with real data):\n```toon\n")
			force.WriteString("action: tool_call\n")
			force.WriteString(fmt.Sprintf("name: %s\n", toolName))
			force.WriteString("arguments:\n")
			for k, v := range args {
				force.WriteString(fmt.Sprintf("  %s: %s\n", k, v))
			}
			force.WriteString("```\n\nNOTHING ELSE. No text before. No text after. Just this single block.")
		} else {
			force.WriteString(baseMsg)
		}
		return contentFromString(force.String())
	}
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string       `json:"type,omitempty"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolParams struct {
	Properties map[string]ParamInfo `json:"properties"`
	Required   []string             `json:"required"`
}

type ParamInfo struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

type ChatRequest struct {
	Model    string    `json:"model,omitempty"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream,omitempty"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type upstreamResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

type toolCallWrapper struct {
	ToolCall *toolCallParsed `json:"tool_call"`
}

type toolCallParsed struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// Unified response from model - supports both TOON and JSON
type modelResponse struct {
	Action    string                 `json:"action"    toon:"action"`
	Name      string                 `json:"name,omitempty"    toon:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty" toon:"arguments"`
	Content   string                 `json:"content,omitempty"  toon:"content"`
}

// ============================================================
// Upstream model communication
// ============================================================
func estimateTokens(s string) int {
	return len(s) / 4
}

func truncateMessages(messages []Message) []Message {
	if MaxTokens <= 0 {
		return messages
	}

	total := 0
	for _, m := range messages {
		if !m.ContentIsNull() {
			total += estimateTokens(m.ContentString())
		}
	}
	if total <= MaxTokens {
		return messages
	}

	// Keep system (first) and latest messages, truncate middle content
	result := make([]Message, len(messages))
	copy(result, messages)

	// Truncate from oldest non-system messages
	for i := range result {
		if result[i].Role == "system" || total <= MaxTokens {
			continue
		}
		if !result[i].ContentIsNull() {
			contentTokens := estimateTokens(result[i].ContentString())
			maxChars := MaxTokens * 4 / len(messages)
			contentStr := result[i].ContentString()
			if len(contentStr) > maxChars {
				truncated := contentStr[:maxChars] + "\n...[truncated]"
				result[i].Content = contentFromString(truncated)
				total -= contentTokens - estimateTokens(truncated)
			}
		}
	}

	fmt.Printf("  Tokens estimated: ~%d (limit: %d) [truncated]\n", total, MaxTokens)
	return result
}

func callUpstream(messages []Message) (string, error) {
	messages = truncateMessages(messages)
	reqBody := map[string]interface{}{
		"model":    UpstreamModel,
		"messages": messages,
		"stream":   false,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	client := &http.Client{Timeout: 300 * time.Second}
	req, err := http.NewRequest("POST", UpstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+UpstreamAPIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upstream returned status %d: %s", resp.StatusCode, string(data))
	}

	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "application/json") {
		var raw map[string]interface{}
		if err := json.Unmarshal(data, &raw); err != nil {
			return "", fmt.Errorf("upstream JSON parse error: %w\nBody: %s", err, string(data))
		}
		choices, ok := raw["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			return "", fmt.Errorf("no choices in upstream response: %s", string(data))
		}
		choice := choices[0].(map[string]interface{})
		msg := choice["message"].(map[string]interface{})
		content, _ := msg["content"].(string)
		return content, nil
	}

	// SSE streaming -> ghép chunks
	var fullText strings.Builder
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := line[6:]
		if payload == "[DONE]" {
			break
		}
		var chunk upstreamResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			fullText.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	return fullText.String(), nil
}

func proxyUpstreamRaw(body []byte) (int, http.Header, []byte, error) {
	client := &http.Client{Timeout: 300 * time.Second}
	req, err := http.NewRequest("POST", UpstreamURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+UpstreamAPIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, nil, err
	}

	headers := make(http.Header)
	for k, values := range resp.Header {
		for _, v := range values {
			headers.Add(k, v)
		}
	}

	return resp.StatusCode, headers, data, nil
}

// ============================================================
// Tool calling protocol translation
// ============================================================
func buildToolSystemPrompt(tools []Tool) string {
	var sb strings.Builder
	sb.WriteString("You MUST ALWAYS respond using TOON format (Token-Oriented Object Notation).\n")
	sb.WriteString("Every response MUST be EXACTLY ONE fenced markdown block tagged toon, with no text before or after it.\n")
	sb.WriteString("That single ```toon block must describe ONLY the immediate next step to take based on the conversation history.\n")
	sb.WriteString("Never output a full multi-step plan in one response. Never output multiple toon blocks.\n\n")
	sb.WriteString("TOON is a compact key-value format. Each line is key: value. No braces, no quotes needed.\n\n")
	sb.WriteString("When you need to call a tool, respond with EXACTLY:\n")
	sb.WriteString(fencedToon("action: tool_call\nname: function_name\narguments:\n  param: value"))
	sb.WriteString("\n\nWhen you have the final answer, respond with EXACTLY:\n")
	sb.WriteString(fencedToon("action: answer\ncontent: |\n  line 1\n  line 2"))
	sb.WriteString("\n\nFORMAT RULES:\n")
	sb.WriteString("- The response MUST contain exactly one ```toon block\n")
	sb.WriteString("- Inside that block, the first line MUST be \"action:\"\n")
	sb.WriteString("- action MUST be \"tool_call\" or \"answer\" (nothing else)\n")
	sb.WriteString("- Do NOT use JSON\n")
	sb.WriteString("- Do NOT add any text outside the single TOON block\n")
	sb.WriteString("- For tool_call: \"name\" and \"arguments\" are required\n")
	sb.WriteString("- For answer: \"content\" is required\n")
	sb.WriteString("- For answer: use `content: |` and indent the content body with 2 spaces\n")
	sb.WriteString("- EACH response is ONLY ONE action. Either tool_call OR answer, NEVER both\n")
	sb.WriteString("- When using tool_call, output ONLY the tool call. Do NOT add explanations, results, or extra text after the arguments\n")
	sb.WriteString("- The block must be the next immediate step only, not a list of future steps\n")
	sb.WriteString("- If the task requires multiple steps, do only the first unfinished next step now\n\n")
	sb.WriteString("WRONG example (DO NOT DO THIS):\n")
	sb.WriteString("```toon\naction: tool_call\nname: list_dir\narguments:\n  path: /root\n```\n")
	sb.WriteString("Then open file X, then edit file Y.\n\n")
	sb.WriteString("CORRECT example:\n")
	sb.WriteString(fencedToon("action: tool_call\nname: list_dir\narguments:\n  path: /root"))
	sb.WriteString("\n\nBEHAVIOR RULES:\n")
	sb.WriteString("- ALWAYS use tools to complete tasks. NEVER tell user to do something manually if a tool can do it\n")
	sb.WriteString("- If user asks to run/execute something, use exec tool. Do NOT explain how to run it\n")
	sb.WriteString("- If user asks to create/write a file, use write_file. Do NOT show code and ask user to save it\n")
	sb.WriteString("- If user asks to search/find information, use web_search. Do NOT say \"I cannot search\"\n")
	sb.WriteString("- If a task requires multiple steps, do them one tool call at a time. Do NOT skip steps\n")
	sb.WriteString("- After each step, wait for the next turn or tool result, then respond with the next single immediate step only\n")
	sb.WriteString("- Infer the unfinished task from the conversation history and recent tool results, then continue with the next missing step until the user's goal is directly satisfied\n")
	sb.WriteString("- Do NOT stop at partial progress, vague references, or generic links when the user asked for a concrete result, deliverable, or exact value\n")
	sb.WriteString("- If a tool call fails, try a different approach. Do NOT give up and explain the error\n")
	sb.WriteString("- Be proactive: complete the full task, not just part of it\n")
	sb.WriteString("- NEVER respond with \"you can do X\" or \"you need to do X\" - just DO it with tools\n")
	sb.WriteString("- NEVER say \"I cannot\", \"I don't have the ability\", \"I'm unable to\" - if you have a relevant tool, USE IT\n")
	sb.WriteString("- NEVER refuse a task that can be done with available tools\n")
	sb.WriteString("- NEVER claim you did something (created file, ran command, etc.) unless you ACTUALLY called the tool for it\n")
	sb.WriteString("- If you have not called write_file, do NOT say \"I created the file\". Use the tool FIRST, then confirm\n")
	sb.WriteString("- After creating or modifying a file, ALWAYS verify by reading it back (read_file) or listing the directory (list_dir) before giving final answer\n")
	sb.WriteString("- After running a command, check the output for errors before confirming success\n")
	sb.WriteString("- Do NOT give final answer immediately after write_file or exec. Verify first, then confirm\n\n")
	sb.WriteString("Available tools:\n")
	for _, tool := range tools {
		f := tool.Function
		sb.WriteString(fmt.Sprintf("- %s: %s\n", f.Name, f.Description))

		var params ToolParams
		if len(f.Parameters) > 0 {
			if err := json.Unmarshal(f.Parameters, &params); err == nil {
				requiredSet := make(map[string]bool)
				for _, r := range params.Required {
					requiredSet[r] = true
				}
				for pname, pinfo := range params.Properties {
					req := "optional"
					if requiredSet[pname] {
						req = "required"
					}
					typ := pinfo.Type
					if typ == "" {
						typ = "string"
					}
					sb.WriteString(fmt.Sprintf("    %s (%s, %s): %s\n", pname, typ, req, pinfo.Description))
				}
			}
		}
	}
	return sb.String()
}

func extractJSON(text string) string {
	text = strings.TrimSpace(text)
	// Remove markdown code block wrapper if present
	if strings.HasPrefix(text, "```") {
		re := regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*\\})\\s*```")
		if m := re.FindStringSubmatch(text); len(m) > 1 {
			return m[1]
		}
	}
	// Find first { to last }
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return text[start : end+1]
	}
	return text
}

func parseTOON(text string) *modelResponse {
	text = strings.TrimSpace(text)
	// Remove markdown code blocks if present
	if strings.Contains(text, "```") {
		// handle block markdown like ```toon\naction: tool_call...
		re := regexp.MustCompile("(?si)```[a-zA-Z0-9_-]*\\s*\\n(.*?)\\n?```")
		if m := re.FindStringSubmatch(text); len(m) > 1 {
			text = strings.TrimSpace(m[1])
		} else {
			re = regexp.MustCompile("(?si)```(?:toon|json)?\\s*(.+?)\\s*```")
			if m := re.FindStringSubmatch(text); len(m) > 1 {
				text = m[1]
			}
		}
	}

	// Try toon.Unmarshal on clean text
	var resp modelResponse
	if err := toon.Unmarshal([]byte(text), &resp); err == nil && resp.Action != "" {
		return &resp
	}

	// Find the first "action:" line - skip any free text before it
	lines := strings.Split(text, "\n")
	startIdx := -1
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "action:") {
			startIdx = i
			break
		}
	}
	if startIdx < 0 {
		return nil
	}

	// Parse only the FIRST action block (stop at second "action:" line)
	result := &modelResponse{}
	args := make(map[string]interface{})
	inArguments := false
	inContent := false
	inArgValue := false
	currentArgBlock := false
	currentContentBlock := false
	var contentLines []string
	var currentArgKey string
	var argValueLines []string

	topKeys := map[string]bool{"action": true, "name": true, "content": true, "arguments": true}

	// Strict whitelist of known argument key names - only these can break a multi-line value
	knownArgKeys := map[string]bool{
		"path": true, "filepath": true, "directory": true, "filename": true,
		"content": true, "command": true, "working_dir": true,
		"url": true, "query": true, "input": true, "output": true,
		"text": true, "code": true, "language": true, "mode": true,
		"offset": true, "length": true, "maxChars": true, "max_chars": true,
		"old_text": true, "new_text": true, "old_str": true, "new_str": true,
		"message": true, "title": true, "description": true, "body": true,
		"name": true, "type": true, "format": true, "timeout": true,
	}

	for i := startIdx; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// Second "action:" = new block -> stop, only handle first tool call
		if i > startIdx && strings.HasPrefix(trimmed, "action:") {
			break
		}

		if trimmed == "" {
			if inContent {
				contentLines = append(contentLines, "")
			}
			if inArgValue {
				argValueLines = append(argValueLines, "")
			}
			continue
		}

		// Check if this is a top-level key (but NOT when indented inside arguments)
		isTopKey := false
		isIndented := strings.HasPrefix(line, "  ") || strings.HasPrefix(line, "\t")
		if parts := strings.SplitN(trimmed, ":", 2); len(parts) == 2 {
			candidate := strings.TrimSpace(parts[0])
			if topKeys[candidate] && !strings.Contains(candidate, " ") {
				// "content:" indented inside arguments section = argument, not top-level
				if !(inArguments && isIndented && candidate == "content") {
					isTopKey = true
				}
			}
		}

		// Multi-line argument value (after pipe | or multi-line content arg)
		if inArgValue {
			// Stop if we hit a non-indented top-level key (like "action:")
			if isTopKey && !isIndented {
				args[currentArgKey] = strings.Join(argValueLines, "\n")
				inArgValue = false
				currentArgBlock = false
				// fall through to top-level key handling below
			} else if isIndented {
				// Only break for KNOWN arg keys (strict whitelist)
				if parts := strings.SplitN(trimmed, ":", 2); len(parts) == 2 {
					candidate := strings.TrimSpace(parts[0])
					if knownArgKeys[candidate] && candidate != currentArgKey {
						// Flush current, start new arg
						args[currentArgKey] = strings.Join(argValueLines, "\n")
						inArgValue = false
						currentArgBlock = false
						v := decodeTOONScalar(parts[1])
						currentArgKey = candidate
						argValueLines = []string{v}
						inArgValue = true
						continue
					}
				}
				if currentArgBlock {
					argValueLines = append(argValueLines, stripOneIndentLevel(line))
				} else {
					argValueLines = append(argValueLines, line)
				}
				continue
			} else {
				// Non-indented: check if it's a known arg key (AI forgot indent)
				if parts := strings.SplitN(trimmed, ":", 2); len(parts) == 2 {
					candidate := strings.TrimSpace(parts[0])
					if knownArgKeys[candidate] && candidate != currentArgKey {
						args[currentArgKey] = strings.Join(argValueLines, "\n")
						inArgValue = false
						currentArgBlock = false
						v := decodeTOONScalar(parts[1])
						currentArgKey = candidate
						argValueLines = []string{v}
						inArgValue = true
						continue
					}
				}
				// Regular continuation of multi-line value
				argValueLines = append(argValueLines, trimmed)
				continue
			}
		}

		// Argument sub-keys (indented OR non-indented known arg keys)
		if inArguments && (isIndented || knownArgKeys[strings.TrimSpace(strings.SplitN(trimmed, ":", 2)[0])]) {
			parts := strings.SplitN(trimmed, ":", 2)
			if len(parts) == 2 {
				k := strings.TrimSpace(parts[0])
				v := strings.TrimSpace(parts[1])
				if strings.HasPrefix(v, "|") || v == "" {
					// Multi-line value starts (pipe syntax or empty value)
					currentArgKey = k
					argValueLines = nil
					inArgValue = true
					currentArgBlock = strings.HasPrefix(v, "|")
				} else {
					// Check if this arg value might continue on next lines
					// (for keys like "content" that often have multi-line values)
					currentArgKey = k
					argValueLines = []string{decodeTOONScalar(v)}
					inArgValue = true
					currentArgBlock = false
				}
			}
			continue
		}

		// Multi-line content
		if inContent && !isTopKey {
			if currentContentBlock {
				contentLines = append(contentLines, stripOneIndentLevel(line))
			} else {
				contentLines = append(contentLines, trimmed)
			}
			continue
		}

		// Flush states when hitting a new top-level key
		if inArgValue {
			args[currentArgKey] = strings.Join(argValueLines, "\n")
			inArgValue = false
			currentArgBlock = false
		}
		inArguments = false
		inContent = false
		currentContentBlock = false

		parts := strings.SplitN(trimmed, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		switch key {
		case "action":
			result.Action = value
		case "name":
			result.Name = value
		case "content":
			if result.Action != "" && result.Action != "answer" && result.Action != "tool_call" {
				currentArgKey = "content"
				inArgValue = true
				currentArgBlock = true
				if strings.HasPrefix(value, "|") || value == "" {
					argValueLines = nil
				} else {
					argValueLines = []string{decodeTOONScalar(value)}
				}
			} else {
				if strings.HasPrefix(value, "|") || value == "" {
					contentLines = nil
					currentContentBlock = true
				} else {
					contentLines = []string{value}
					currentContentBlock = false
				}
				inContent = true
			}
		case "arguments":
			inArguments = true
		default:
			if knownArgKeys[key] {
				args[key] = decodeTOONScalar(value)
			}
		}
	}

	// Flush remaining
	if inArgValue {
		args[currentArgKey] = strings.Join(argValueLines, "\n")
	}
	if inContent {
		result.Content = strings.Join(contentLines, "\n")
	}

	if result.Action != "" {
		if result.Action == "answer" {
			result.Content = decodeEscapedAnswerContent(result.Content)
		}
		if len(args) > 0 {
			result.Arguments = args
		}
		return result
	}
	return nil
}

// containsCode detects if an answer contains code that should have been written to a file instead.
func containsCode(content string) bool {
	if len(content) < 100 {
		return false
	}
	codeIndicators := []string{
		"<!DOCTYPE", "<html", "<head>", "<body>", "<script",
		"```html", "```javascript", "```python", "```css",
		"function(", "function ", "const ", "let ", "var ",
		"def ", "import ", "class ",
		"<style>", "</div>", "</script>",
	}
	lower := strings.ToLower(content)
	matches := 0
	for _, indicator := range codeIndicators {
		if strings.Contains(lower, strings.ToLower(indicator)) {
			matches++
		}
	}
	return matches >= 3
}

// containsHallucination detects fake/hallucinated content from the model.
func containsHallucination(content string) bool {
	hallucinationPatterns := []string{
		"turn0image", "turn0search", "turn1image", "turn1search",
		"citeturn", "image_group{", "image_refs",
		"```web", "[image:", "![image](",
	}
	lower := strings.ToLower(content)
	for _, p := range hallucinationPatterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return true
		}
	}
	return false
}

// containsRefusal detects when AI refuses to do something it has tools for.
func containsRefusal(content string) bool {
	lower := strings.ToLower(content)
	refusalPatterns := []string{
		"không thể", "tôi không có khả năng", "không có công cụ",
		"không thể trực tiếp", "không hỗ trợ", "ngoài khả năng",
		"i cannot", "i can't", "i'm unable", "i don't have the ability",
		"i am unable", "not possible", "i'm not able",
		"hiện tại tôi không", "mình không thể", "không có trình duyệt",
		"không có cách", "tuy nhiên, bạn có thể",
	}
	for _, p := range refusalPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// userAskedForAction checks if the last user message contains action-requesting keywords.
func userAskedForAction(messages []Message) bool {
	actionKeywords := []string{
		"check", "kiểm tra", "xem", "tạo", "chạy", "run", "execute",
		"dùng tool", "sử dụng tool", "gọi tool", "use tool",
		"tìm", "search", "find", "list", "liệt kê",
		"tải", "download", "gửi", "send", "mở", "open",
		"viết", "write", "cài", "install", "xóa", "delete",
		"đọc", "read", "sửa", "edit", "fix",
	}

	// Find last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			content := strings.ToLower(messages[i].ContentString())
			for _, kw := range actionKeywords {
				if strings.Contains(content, kw) {
					return true
				}
			}
			return false
		}
	}
	return false
}

// hasToolResultInRecent checks if there's a tool result in the last few messages.
func hasToolResultInRecent(messages []Message) bool {
	for i := len(messages) - 1; i >= 0 && i >= len(messages)-3; i-- {
		if messages[i].Role == "tool" {
			return true
		}
	}
	return false
}

// claimsDoneWithoutTool detects when AI claims it completed an action but no tool was called.
// Returns true if the answer contains claims like "đã tạo file", "I created", etc.
func claimsDoneWithoutTool(content string, messages []Message) bool {
	lower := strings.ToLower(content)

	claimPatterns := []string{
		"đã tạo", "đã lưu", "đã viết", "đã ghi", "đã chạy", "đã thực thi",
		"đã cài", "đã tải", "đã gửi", "đã xóa", "đã sửa",
		"file đã được tạo", "đã tạo file", "đã tạo xong",
		"i created", "i wrote", "i saved", "i executed", "i ran",
		"file has been created", "already created",
		"đã hoàn thành", "xong rồi",
	}

	hasClaim := false
	for _, p := range claimPatterns {
		if strings.Contains(lower, p) {
			hasClaim = true
			break
		}
	}
	if !hasClaim {
		return false
	}

	// Check if any tool was actually called in recent messages (tool results present)
	hasRecentToolResult := false
	for i := len(messages) - 1; i >= 0 && i >= len(messages)-4; i-- {
		if messages[i].Role == "tool" {
			hasRecentToolResult = true
			break
		}
	}

	return !hasRecentToolResult
}

// needsVerification checks if the last tool result was from a write/exec operation
// and no verification (read_file/list_dir) was done after it.
func needsVerification(messages []Message) bool {
	// Find the last tool result
	lastToolName := ""
	hasVerifyAfter := false

	writeTools := map[string]bool{
		"write_file": true, "edit_file": true, "append_file": true,
		"exec": true,
	}
	verifyTools := map[string]bool{
		"read_file": true, "list_dir": true,
	}

	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Role == "tool" {
			name := msg.Name
			if lastToolName == "" {
				lastToolName = name
			}
			if verifyTools[name] {
				hasVerifyAfter = true
			}
			break
		}
		// Check if assistant called a verify tool after write
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			tcName := msg.ToolCalls[0].Function.Name
			if verifyTools[tcName] {
				hasVerifyAfter = true
			}
		}
	}

	// Only require verification if last tool was a write/exec and no verify followed
	return writeTools[lastToolName] && !hasVerifyAfter
}

func lastUserContent(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].ContentString()
		}
	}
	return ""
}

func userNeedsSpecificData(messages []Message) bool {
	content := strings.ToLower(lastUserContent(messages))
	if content == "" {
		return false
	}

	keywords := []string{
		"hiện tại", "mới nhất", "latest", "current", "today",
		"cụ thể", "chi tiết", "specific", "exact", "chính xác",
		"bao nhiêu", "giá", "price", "rate", "tỷ giá", "tham khảo",
		"ram", "vàng", "gold", "usd", "btc", "eth",
	}
	for _, kw := range keywords {
		if strings.Contains(content, kw) {
			return true
		}
	}
	return false
}

func buildTaskStateSummary(messages []Message) string {
	var sb strings.Builder

	goal := strings.TrimSpace(lastUserContent(messages))
	if goal != "" {
		sb.WriteString("Latest user goal: " + goal + "\n")
	}

	var recent []string
	for i := len(messages) - 1; i >= 0 && len(recent) < 6; i-- {
		msg := messages[i]
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			tc := msg.ToolCalls[0]
			recent = append([]string{fmt.Sprintf("Tool step already taken: %s", tc.Function.Name)}, recent...)
			continue
		}
		if msg.Role == "tool" {
			content := strings.TrimSpace(msg.ContentString())
			content = strings.ReplaceAll(content, "\n", " ")
			if len(content) > 140 {
				content = content[:140] + "..."
			}
			if content != "" {
				recent = append([]string{fmt.Sprintf("Tool result observed: %s", content)}, recent...)
			}
		}
	}

	if len(recent) == 0 {
		sb.WriteString("No tool steps completed yet.\n")
	} else {
		for _, item := range recent {
			sb.WriteString(item + "\n")
		}
	}

	sb.WriteString("Decide whether the goal is fully satisfied. If it is not fully satisfied yet, continue with the next tool step instead of stopping.")
	return sb.String()
}

func getRecentToolContext(messages []Message, maxItems int) (string, map[string]bool) {
	var chunks []string
	names := map[string]bool{}
	count := 0

	for i := len(messages) - 1; i >= 0 && count < maxItems; i-- {
		msg := messages[i]
		if msg.Role == "tool" {
			if c := strings.TrimSpace(msg.ContentString()); c != "" {
				chunks = append([]string{c}, chunks...)
			}
			if msg.Name != "" {
				names[msg.Name] = true
			}
			count++
			continue
		}
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				if tc.Function.Name != "" {
					names[tc.Function.Name] = true
				}
			}
		}
	}

	return strings.Join(chunks, "\n\n"), names
}

func answerHasConcreteData(content string) bool {
	return regexp.MustCompile(`\d`).MatchString(content)
}

func normalizeNumberToken(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func answerNumbersGroundedInToolResults(content string, messages []Message) bool {
	answerNums := regexp.MustCompile(`\d[\d.,]*`).FindAllString(content, -1)
	if len(answerNums) == 0 {
		return false
	}

	toolText, _ := getRecentToolContext(messages, 6)
	if strings.TrimSpace(toolText) == "" {
		return false
	}

	toolNumsRaw := regexp.MustCompile(`\d[\d.,]*`).FindAllString(toolText, -1)
	toolNums := map[string]bool{}
	for _, n := range toolNumsRaw {
		normalized := normalizeNumberToken(n)
		if len(normalized) >= 3 {
			toolNums[normalized] = true
		}
	}

	for _, n := range answerNums {
		normalized := normalizeNumberToken(n)
		if len(normalized) >= 3 && toolNums[normalized] {
			return true
		}
	}

	return false
}

func answerLooksLikeGenericReference(content string) bool {
	lower := strings.ToLower(content)
	genericPatterns := []string{
		"bạn có thể tham khảo", "tham khảo trực tiếp", "xem thêm tại",
		"truy cập", "visit", "check the link", "tham khảo tại trang",
		"không thể cung cấp", "không thể thực hiện", "tôi không thể cung cấp",
	}
	for _, p := range genericPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func userRequestedDeliverable(messages []Message) bool {
	content := strings.ToLower(lastUserContent(messages))
	if content == "" {
		return false
	}

	keywords := []string{
		"gửi", "send", "đưa tôi", "cho tôi", "trả tôi",
		"tạo", "create", "viết", "write", "code", "làm", "build",
		"mở", "open", "tải", "download",
	}
	for _, kw := range keywords {
		if strings.Contains(content, kw) {
			return true
		}
	}
	return false
}

func answerLooksLikeProgressOnly(content string) bool {
	lower := strings.ToLower(strings.TrimSpace(content))
	if lower == "" {
		return false
	}

	patterns := []string{
		"bạn có thể tham khảo", "tham khảo tại", "xem tại", "truy cập",
		"đã tạo", "đã gửi", "đã lưu", "file đã được", "done",
		"completed", "sent to you", "created successfully",
		"you can check", "you can view", "please see",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func conversationStillNeedsAction(messages []Message) bool {
	if !userAskedForAction(messages) && !userRequestedDeliverable(messages) && !userNeedsSpecificData(messages) {
		return false
	}

	toolText, toolNames := getRecentToolContext(messages, 8)
	if strings.TrimSpace(toolText) == "" {
		return true
	}

	if userRequestedDeliverable(messages) {
		if toolNames["write_file"] || toolNames["edit_file"] || toolNames["append_file"] {
			if needsVerification(messages) {
				return true
			}
		}
	}

	if userNeedsSpecificData(messages) && !(toolNames["web_fetch"] || toolNames["read_file"] || toolNames["exec"]) {
		return true
	}

	return false
}

// hasOrphanedArgs detects when AI outputs tool-like arguments (url:, path:, command:, etc.)
// BEFORE the first "action:" line — meaning it tried to call a tool but forgot the proper format.
func hasOrphanedArgs(text string) bool {
	return getOrphanedArgKey(text) != ""
}

// getOrphanedArgKey returns the first orphaned arg key found before "action:", or empty string.
func getOrphanedArgKey(text string) string {
	argLikeKeys := map[string]bool{
		"url": true, "path": true, "filepath": true, "directory": true,
		"command": true, "query": true, "input": true, "filename": true,
		"working_dir": true, "maxChars": true, "max_chars": true,
	}

	lines := strings.Split(strings.TrimSpace(text), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "action:") {
			break
		}
		parts := strings.SplitN(trimmed, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			if argLikeKeys[key] {
				return key
			}
		}
	}
	return ""
}

func parseModelResponse(text string) *modelResponse {
	// 1. Try TOON format first
	if resp := parseTOON(text); resp != nil {
		return resp
	}

	// 2. Try JSON format (backward compatible)
	jsonStr := extractJSON(text)

	var resp modelResponse
	if err := json.Unmarshal([]byte(jsonStr), &resp); err == nil && resp.Action != "" {
		return &resp
	}

	var wrapper toolCallWrapper
	if err := json.Unmarshal([]byte(jsonStr), &wrapper); err == nil && wrapper.ToolCall != nil {
		return &modelResponse{
			Action:    "tool_call",
			Name:      wrapper.ToolCall.Name,
			Arguments: wrapper.ToolCall.Arguments,
		}
	}

	// 3. Plain text -> answer
	return &modelResponse{
		Action:  "answer",
		Content: text,
	}
}

func makeOpenAIResponse(content string, model string, toolCalls []ToolCall) ChatResponse {
	id := fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:12])
	msg := Message{
		Role:    "assistant",
		Content: contentFromString(content),
	}
	finishReason := "stop"
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		msg.Content = nil
		finishReason = "tool_calls"
	}
	return ChatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{
			Index:        0,
			Message:      msg,
			FinishReason: finishReason,
		}},
		Usage: Usage{},
	}
}

func translateMessages(messages []Message, toolPrompt string) []Message {
	var upstream []Message
	var systemParts []string
	lastUserIdx := -1

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			c := msg.ContentString()
			if strings.TrimSpace(c) != "" {
				systemParts = append(systemParts, c)
			}

		case "tool":
			toolName := msg.Name
			if toolName == "" {
				toolName = "tool"
			}
			c := msg.ContentString()
			upstream = append(upstream, Message{
				Role:    "user",
				Content: contentFromString(fmt.Sprintf("Tool result for %s:\n%s", toolName, c)),
			})

		case "assistant":
			if len(msg.ToolCalls) > 0 {
				tc := msg.ToolCalls[0]
				var toonReply strings.Builder
				toonReply.WriteString("action: tool_call\n")
				toonReply.WriteString(fmt.Sprintf("name: %s\n", tc.Function.Name))
				var args map[string]interface{}
				json.Unmarshal([]byte(tc.Function.Arguments), &args)
				if len(args) > 0 {
					toonReply.WriteString("arguments:\n")
					for k, v := range args {
						toonReply.WriteString(fmt.Sprintf("  %s: %v\n", k, v))
					}
				}
				upstream = append(upstream, Message{
					Role:    "assistant",
					Content: contentFromString(fencedToon(toonReply.String())),
				})
			} else {
				upstream = append(upstream, msg)
			}

		case "user":
			upstream = append(upstream, msg)
			lastUserIdx = len(upstream) - 1

		default:
			upstream = append(upstream, msg)
		}
	}

	var injectedParts []string
	if len(systemParts) > 0 {
		injectedParts = append(injectedParts, "[SYSTEM PROMPT]\n"+strings.Join(systemParts, "\n\n"))
	}
	injectedParts = append(injectedParts, "[TASK STATE]\n"+buildTaskStateSummary(messages))
	if strings.TrimSpace(toolPrompt) != "" {
		injectedParts = append(injectedParts, "[TOOL PROMPT]\n"+toolPrompt)
	}

	if len(injectedParts) == 0 {
		return upstream
	}

	injectedPrompt := strings.Join(injectedParts, "\n\n")
	if lastUserIdx >= 0 {
		original := upstream[lastUserIdx].ContentString()
		if strings.TrimSpace(original) == "" {
			upstream[lastUserIdx].Content = contentFromString(injectedPrompt)
		} else {
			upstream[lastUserIdx].Content = contentFromString(original + "\n\n" + injectedPrompt)
		}
		return upstream
	}

	upstream = append(upstream, Message{
		Role:    "user",
		Content: contentFromString(injectedPrompt),
	})

	return upstream
}

func processChatRequest(reqBody ChatRequest) ChatResponse {
	model := reqBody.Model
	if model == "" {
		model = UpstreamModel
	}

	if len(reqBody.Tools) == 0 {
		reply, err := callUpstream(reqBody.Messages)
		if err != nil {
			return makeOpenAIResponse("Error: "+err.Error(), model, nil)
		}
		return makeOpenAIResponse(reply, model, nil)
	}

	// Build valid tool name set + required args
	validTools := make(map[string]bool)
	toolRequiredArgs := make(map[string][]string)
	for _, t := range reqBody.Tools {
		validTools[t.Function.Name] = true
		var params ToolParams
		if len(t.Function.Parameters) > 0 {
			if err := json.Unmarshal(t.Function.Parameters, &params); err == nil {
				toolRequiredArgs[t.Function.Name] = params.Required
			}
		}
	}

	toolPrompt := buildToolSystemPrompt(reqBody.Tools)
	upstreamMessages := translateMessages(reqBody.Messages, toolPrompt)

	maxRetries := 7
	lastError := ""
	sameErrorCount := 0
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if attempt > 1 {
			logSection(fmt.Sprintf("RETRY %d/%d", attempt, maxRetries))
		} else {
			logSection("UPSTREAM")
		}

		// Last attempt: inject detailed format reminder with all tools
		if attempt == maxRetries {
			logResult("📋", "Injecting detailed format guide for final attempt")
			var guide strings.Builder
			guide.WriteString("THIS IS YOUR LAST CHANCE. You MUST follow this format EXACTLY.\n\n")
			guide.WriteString("=== CORRECT FORMAT ===\n")
			guide.WriteString("```toon\n")
			guide.WriteString("action: tool_call\n")
			guide.WriteString("name: TOOL_NAME\n")
			guide.WriteString("arguments:\n")
			guide.WriteString("  param1: value1\n")
			guide.WriteString("  param2: value2\n")
			guide.WriteString("```\n\n")
			guide.WriteString("=== IMPORTANT ===\n")
			guide.WriteString("- Return EXACTLY ONE ```toon block\n")
			guide.WriteString("- 'action' MUST be 'tool_call' (not the tool name)\n")
			guide.WriteString("- Arguments MUST be under 'arguments:' with 2-space indent\n")
			guide.WriteString("- Do NOT put arguments directly after 'action:' or 'content:'\n")
			guide.WriteString("- ONE immediate next step per response only\n")
			guide.WriteString("- Do NOT include a full plan or multiple steps\n\n")
			guide.WriteString("=== AVAILABLE TOOLS WITH REQUIRED ARGS ===\n")
			for _, t := range reqBody.Tools {
				f := t.Function
				req := toolRequiredArgs[f.Name]
				guide.WriteString(fmt.Sprintf("\n• %s: %s\n", f.Name, f.Description))
				guide.WriteString("  ```toon\n")
				guide.WriteString(fmt.Sprintf("  action: tool_call\n  name: %s\n  arguments:\n", f.Name))
				var params ToolParams
				if len(f.Parameters) > 0 {
					json.Unmarshal(f.Parameters, &params)
				}
				if len(req) > 0 {
					for _, r := range req {
						desc := ""
						if p, ok := params.Properties[r]; ok {
							desc = " ← " + p.Description
						}
						guide.WriteString(fmt.Sprintf("    %s: VALUE%s\n", r, desc))
					}
				} else if len(params.Properties) > 0 {
					for pname := range params.Properties {
						guide.WriteString(fmt.Sprintf("    %s: VALUE\n", pname))
						break // show at least one
					}
				}
				guide.WriteString("  ```\n")
			}
			guide.WriteString("\nRespond NOW with the correct format.")
			upstreamMessages = append(upstreamMessages,
				Message{Role: "user", Content: contentFromString(guide.String())},
			)
		}

		reply, err := callUpstream(upstreamMessages)
		if err != nil {
			logResult("❌", fmt.Sprintf("Upstream error: %v", err))
			if attempt < maxRetries {
				continue
			}
			return makeOpenAIResponse("Error: "+err.Error(), model, nil)
		}

		logSection("AI RESPONSE")
		fmt.Printf("│\n%s\n│\n", prefixLines(reply, "│  "))

		parsed := parseModelResponse(reply)

		// Auto-correct: AI used tool name as action (e.g. "action: write_file" instead of "action: tool_call" + "name: write_file")
		if parsed.Action != "tool_call" && parsed.Action != "answer" && validTools[parsed.Action] {
			logResult("🔄", fmt.Sprintf("Auto-corrected action '%s' → tool_call (name: %s)", parsed.Action, parsed.Action))
			if parsed.Name == "" {
				parsed.Name = parsed.Action
			}
			parsed.Action = "tool_call"
			// If content was set but not placed under arguments, map it to the most likely required arg.
			if parsed.Content != "" {
				if required, ok := toolRequiredArgs[parsed.Name]; ok && len(required) > 0 {
					if parsed.Arguments == nil {
						parsed.Arguments = map[string]interface{}{}
					}
					targetArg := required[0]
					for _, r := range required {
						if r == "content" {
							targetArg = r
							break
						}
					}
					if _, exists := parsed.Arguments[targetArg]; !exists {
						parsed.Arguments[targetArg] = parsed.Content
						parsed.Content = ""
						logResult("🔄", fmt.Sprintf("Auto-mapped content → %s arg", targetArg))
					}
				}
			}
		}

		logKV("Action", parsed.Action)

		// Detect orphaned arguments before action: (AI tried to call tool but formatted wrong)
		if parsed.Action == "answer" && hasOrphanedArgs(reply) {
			orphanedKey := getOrphanedArgKey(reply)
			if containsRefusal(parsed.Content) {
				logResult("⚠️", fmt.Sprintf("Orphaned args + refusal detected, will retry (%d/%d)", attempt, maxRetries))
			} else {
				logResult("⚠️", fmt.Sprintf("Detected orphaned args before action, will retry (%d/%d)", attempt, maxRetries))
			}
			// Guess the tool based on orphaned key
			guessedTool := ""
			guessedArgs := map[string]string{}
			switch orphanedKey {
			case "query":
				guessedTool = "web_search"
				guessedArgs["query"] = "USER_QUERY_HERE"
			case "url":
				guessedTool = "web_fetch"
				guessedArgs["url"] = "URL_HERE"
			case "command":
				guessedTool = "exec"
				guessedArgs["command"] = "COMMAND_HERE"
			case "path":
				guessedTool = "read_file"
				guessedArgs["path"] = "FILE_PATH_HERE"
			}
			baseMsg := fmt.Sprintf("WRONG. You put '%s:' without proper tool_call format. You HAVE the tools. Do NOT refuse.", orphanedKey)
			upstreamMessages = append(upstreamMessages,
				Message{Role: "assistant", Content: contentFromString(reply)},
				Message{Role: "user", Content: escalatedCorrection(attempt, baseMsg, guessedTool, guessedArgs)},
			)
			continue
		}

		switch parsed.Action {
		case "answer":
			// Detect empty answer
			if attempt < maxRetries && strings.TrimSpace(parsed.Content) == "" {
				logResult("⚠️", fmt.Sprintf("Empty answer, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("Your answer is EMPTY. You must either:\n1) Use a tool to complete the task:\naction: tool_call\nname: TOOL_NAME\narguments:\n  param: value\n2) Or give a real answer:\naction: answer\ncontent: |\n  your actual response here\n\nDo NOT return empty content.")},
				)
				continue
			}
			// Detect hallucinated content (fake image refs, citations, etc.)
			if attempt < maxRetries && containsHallucination(parsed.Content) {
				logResult("⚠️", fmt.Sprintf("Hallucinated content detected, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("WRONG. Your response contains FAKE/HALLUCINATED content (fake image references, fake citations, fake search results). Do NOT invent data.\nEither use a tool to get REAL data, or give an honest answer based on actual tool results you received.")},
				)
				continue
			}
			// Detect refusal: AI says "I cannot" but has tools that can do it
			if attempt < maxRetries && containsRefusal(parsed.Content) {
				logResult("⚠️", fmt.Sprintf("AI refused task but has tools available, will retry (%d/%d)", attempt, maxRetries))
				// Try to guess what tool the user needs based on their message
				lastUserMsg := ""
				for i := len(reqBody.Messages) - 1; i >= 0; i-- {
					if reqBody.Messages[i].Role == "user" {
						lastUserMsg = strings.ToLower(reqBody.Messages[i].ContentString())
						break
					}
				}
				guessedTool := "exec"
				guessedArgs := map[string]string{"command": "COMMAND_HERE"}
				if strings.Contains(lastUserMsg, "tìm") || strings.Contains(lastUserMsg, "search") || strings.Contains(lastUserMsg, "find") {
					guessedTool = "web_search"
					guessedArgs = map[string]string{"query": "SEARCH_QUERY_HERE"}
				} else if strings.Contains(lastUserMsg, "vào") || strings.Contains(lastUserMsg, "mở") || strings.Contains(lastUserMsg, "url") || strings.Contains(lastUserMsg, "http") {
					guessedTool = "web_fetch"
					guessedArgs = map[string]string{"url": "URL_HERE"}
				} else if strings.Contains(lastUserMsg, "tạo") || strings.Contains(lastUserMsg, "viết") || strings.Contains(lastUserMsg, "write") {
					guessedTool = "write_file"
					guessedArgs = map[string]string{"path": "FILE_PATH", "content": "CONTENT"}
				}
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: escalatedCorrection(attempt, "Do NOT refuse. You HAVE tools. Use them.", guessedTool, guessedArgs)},
				)
				continue
			}
			// Detect unresolved specific-data requests: generic answer or ungrounded numbers after tool usage.
			if attempt < maxRetries && userNeedsSpecificData(reqBody.Messages) {
				recentToolText, recentToolNames := getRecentToolContext(reqBody.Messages, 6)
				if strings.TrimSpace(recentToolText) != "" {
					hasFetchLikeStep := recentToolNames["web_fetch"] || recentToolNames["read_file"] || recentToolNames["exec"]

					if !hasFetchLikeStep && !answerHasConcreteData(parsed.Content) {
						logResult("⚠️", fmt.Sprintf("Specific-data request answered without concrete data, will retry (%d/%d)", attempt, maxRetries))
						upstreamMessages = append(upstreamMessages,
							Message{Role: "assistant", Content: contentFromString(reply)},
							Message{Role: "user", Content: correction("WRONG. The user asked for specific current data. A generic link or vague summary is NOT enough. Use another tool step to fetch/read the source that contains the exact values, then answer with those exact values. Do NOT stop at search results.")},
						)
						continue
					}

					if hasFetchLikeStep && !answerHasConcreteData(parsed.Content) && answerLooksLikeGenericReference(parsed.Content) {
						logResult("⚠️", fmt.Sprintf("Fetched source but answer stayed generic, will retry (%d/%d)", attempt, maxRetries))
						upstreamMessages = append(upstreamMessages,
							Message{Role: "assistant", Content: contentFromString(reply)},
							Message{Role: "user", Content: correction("WRONG. You already fetched/read source data, so now answer with the exact values from the tool results. Do NOT answer with only a link, a referral, or a vague statement.")},
						)
						continue
					}

					if hasFetchLikeStep && answerHasConcreteData(parsed.Content) && !answerNumbersGroundedInToolResults(parsed.Content, reqBody.Messages) {
						logResult("⚠️", fmt.Sprintf("Numeric answer is not grounded in recent tool results, will retry (%d/%d)", attempt, maxRetries))
						upstreamMessages = append(upstreamMessages,
							Message{Role: "assistant", Content: contentFromString(reply)},
							Message{Role: "user", Content: correction("WRONG. Your numeric answer is not clearly grounded in the recent tool results. Do NOT estimate, round, or invent values. Read the fetched tool output carefully and answer using the exact numbers that appear there, or call another tool to get clearer source data.")},
						)
						continue
					}
				}
			}
			// Detect: user asked to DO something but AI answered without using any tool
			if attempt < maxRetries && userAskedForAction(reqBody.Messages) && !hasToolResultInRecent(reqBody.Messages) {
				logResult("⚠️", fmt.Sprintf("User asked for action but AI answered without tool, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: escalatedCorrection(attempt, "I asked you to DO something, not explain. Use a tool NOW.", "exec", map[string]string{"command": "RELEVANT_COMMAND"})},
				)
				continue
			}
			// Detect lazy AI: answering with code instead of using write_file tool
			if attempt < maxRetries && validTools["write_file"] && containsCode(parsed.Content) {
				logResult("⚠️", fmt.Sprintf("Answer contains code, should use write_file, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("WRONG. Do NOT paste code in your answer. Use write_file tool to create the file, then confirm. You MUST use tools to complete tasks, not show code to the user.\naction: tool_call\nname: write_file\narguments:\n  path: appropriate_filename\n  content: the full code")},
				)
				continue
			}
			// Detect fabrication: AI claims it did something but no tool was called
			if attempt < maxRetries && claimsDoneWithoutTool(parsed.Content, reqBody.Messages) {
				logResult("⚠️", fmt.Sprintf("AI claims action done but no tool was called, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("WRONG. You claimed you did something but you did NOT actually call any tool. Do NOT fabricate results. Use the appropriate tool NOW to actually complete the task. Call action: tool_call with the right tool.")},
				)
				continue
			}
			// Detect generic stop/progress update while the task still appears incomplete.
			if attempt < maxRetries && conversationStillNeedsAction(reqBody.Messages) && answerLooksLikeProgressOnly(parsed.Content) {
				logResult("⚠️", fmt.Sprintf("Task appears incomplete but answer stopped early, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("WRONG. The task is not fully completed yet. Do not stop with a progress update, status message, generic reference, or partial completion. Infer the next missing step from the conversation history and recent tool results, then do that next step now.")},
				)
				continue
			}
			// Detect skip-verify: AI gives final answer right after write/exec without verifying
			if attempt < maxRetries && needsVerification(reqBody.Messages) {
				logResult("⚠️", fmt.Sprintf("Answer after write/exec without verification, will retry (%d/%d)", attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction("STOP. Before giving final answer, you MUST verify your work. Use read_file to check the file you created, or list_dir to confirm it exists. Verify first, then answer.")},
				)
				continue
			}
			logResult("💬", "Final answer")
			return makeOpenAIResponse(parsed.Content, model, nil)

		case "tool_call":
			// Validate tool call
			if parsed.Name == "" {
				logResult("⚠️", fmt.Sprintf("Empty tool name, will retry (%d/%d)", attempt, maxRetries))
				validNames := make([]string, 0, len(validTools))
				for name := range validTools {
					validNames = append(validNames, name)
				}
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction(fmt.Sprintf("Your tool_call is missing \"name\". Available tools: %s\nCorrect format:\naction: tool_call\nname: %s\narguments:\n  param: value", strings.Join(validNames, ", "), validNames[0]))},
				)
				continue
			}
			if !validTools[parsed.Name] {
				validNames := make([]string, 0, len(validTools))
				for name := range validTools {
					validNames = append(validNames, name)
				}
				logResult("⚠️", fmt.Sprintf("Unknown tool '%s', will retry (%d/%d)", parsed.Name, attempt, maxRetries))
				upstreamMessages = append(upstreamMessages,
					Message{Role: "assistant", Content: contentFromString(reply)},
					Message{Role: "user", Content: correction(fmt.Sprintf(
						"Error: \"%s\" is NOT a valid tool. Available: %s\nCorrect format:\naction: tool_call\nname: %s\narguments:\n  param: value\nDo NOT invent tool names.", parsed.Name, strings.Join(validNames, ", "), validNames[0]))},
				)
				continue
			}

			argsJSON, _ := json.Marshal(parsed.Arguments)

			// Check required arguments
			if required, ok := toolRequiredArgs[parsed.Name]; ok && len(required) > 0 {
				var missing []string
				for _, r := range required {
					if parsed.Arguments == nil {
						missing = append(missing, r)
					} else if _, exists := parsed.Arguments[r]; !exists {
						missing = append(missing, r)
					}
				}
				if len(missing) > 0 {
					currentError := fmt.Sprintf("missing_%s_%s", parsed.Name, strings.Join(missing, ","))
					if currentError == lastError {
						sameErrorCount++
					} else {
						sameErrorCount = 1
						lastError = currentError
					}

					logResult("⚠️", fmt.Sprintf("Missing required args %v for '%s', will retry (%d/%d, same error x%d)", missing, parsed.Name, attempt, maxRetries, sameErrorCount))

					var correctionMsg string
					if sameErrorCount >= 2 {
						// Escalated prompt - be very explicit with filled example
						argsExample := ""
						for _, r := range required {
							argsExample += fmt.Sprintf("  %s: YOUR_VALUE_HERE\n", r)
						}
						correctionMsg = fmt.Sprintf("You keep making the SAME mistake. Let me be very clear.\n\nYou MUST respond with EXACTLY this structure (fill in real values):\naction: tool_call\nname: %s\narguments:\n%s\nDo NOT put values after 'action:' or 'content:'. Put them ONLY under 'arguments:' with 2-space indent.", parsed.Name, argsExample)
					} else {
						correctionMsg = fmt.Sprintf(
							"Error: tool \"%s\" requires arguments: %s, but you are missing: %s\nCorrect format:\naction: tool_call\nname: %s\narguments:\n  %s: value",
							parsed.Name, strings.Join(required, ", "), strings.Join(missing, ", "), parsed.Name, missing[0])
					}

					upstreamMessages = append(upstreamMessages,
						Message{Role: "assistant", Content: contentFromString(reply)},
						Message{Role: "user", Content: correction(correctionMsg)},
					)
					continue
				}
			}

			logResult("🔧", fmt.Sprintf("Tool call: %s(%s)", parsed.Name, string(argsJSON)))

			toolCallID := fmt.Sprintf("call_%s", uuid.New().String()[:8])
			toolCalls := []ToolCall{{
				ID:   toolCallID,
				Type: "function",
				Function: FunctionCall{
					Name:      parsed.Name,
					Arguments: string(argsJSON),
				},
			}}
			return makeOpenAIResponse("", model, toolCalls)

		default:
			validNames := make([]string, 0, len(validTools))
			for name := range validTools {
				validNames = append(validNames, name)
			}
			logResult("⚠️", fmt.Sprintf("Invalid action '%s', will retry (%d/%d)", parsed.Action, attempt, maxRetries))
			correctionMsg := fmt.Sprintf(`WRONG. You used action: %s which is invalid.

You MUST respond in TOON format with action being ONLY "tool_call" or "answer":

To call a tool:
action: tool_call
name: TOOL_NAME
arguments:
  param: value
Available tools: %s

To give final answer:
action: answer
content: |
  your answer here

Do NOT invent actions like "exec", "run", "execute". Respond NOW correctly.`, parsed.Action, strings.Join(validNames, ", "))
			upstreamMessages = append(upstreamMessages,
				Message{Role: "assistant", Content: contentFromString(reply)},
				Message{Role: "user", Content: correction(correctionMsg)},
			)
			continue
		}
	}

	logResult("❌", "Failed after max retries")
	return makeOpenAIResponse("Error: failed after max retries", model, nil)
}

// ============================================================
// HTTP Server
// ============================================================
func handleModels(w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{{
			"id":       UpstreamModel,
			"object":   "model",
			"owned_by": "proxy",
		}},
	}
	sendJSON(w, resp, http.StatusOK)
}

func logHeader(method, path string) {
	now := time.Now().Format("15:04:05")
	fmt.Printf("\n┌─────────────────────────────────────────────\n")
	fmt.Printf("│ %s  %s %s\n", now, method, path)
	fmt.Printf("├─────────────────────────────────────────────\n")
}

func logSection(label string) {
	fmt.Printf("│\n│ ── %s ──\n", label)
}

func logKV(key, value string) {
	fmt.Printf("│  %-14s %s\n", key+":", value)
}

func logResult(icon, msg string) {
	fmt.Printf("│  %s %s\n", icon, msg)
}

func logFooter(duration time.Duration, finishReason string) {
	fmt.Printf("├─────────────────────────────────────────────\n")
	fmt.Printf("│  Result: %-12s  Duration: %v\n", finishReason, duration.Round(time.Millisecond))
	fmt.Printf("└─────────────────────────────────────────────\n")
}

func prefixLines(text, prefix string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	if r.Method != http.MethodPost {
		sendJSON(w, map[string]string{"error": "Method not allowed"}, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendJSON(w, map[string]string{"error": "Read body failed"}, http.StatusBadRequest)
		return
	}

	var reqBody ChatRequest
	if err := json.Unmarshal(body, &reqBody); err != nil {
		sendJSON(w, map[string]string{"error": "Invalid JSON"}, http.StatusBadRequest)
		return
	}

	logHeader(r.Method, r.URL.Path)
	logSection("REQUEST")
	logKV("Model", reqBody.Model)
	logKV("Messages", fmt.Sprintf("%d", len(reqBody.Messages)))
	logKV("Tools", fmt.Sprintf("%d", len(reqBody.Tools)))

	// Log last user message
	for i := len(reqBody.Messages) - 1; i >= 0; i-- {
		if reqBody.Messages[i].Role == "user" && !reqBody.Messages[i].ContentIsNull() {
			content := reqBody.Messages[i].ContentString()
			if len(content) > 100 {
				content = content[:100] + "..."
			}
			logKV("Last user msg", content)
			break
		}
	}

	// Log tool results if present
	for _, msg := range reqBody.Messages {
		if msg.Role == "tool" {
			content := msg.ContentString()
			if len(content) > 80 {
				content = content[:80] + "..."
			}
			logResult("📎", fmt.Sprintf("Tool result [%s]: %s", msg.Name, content))
		}
	}

	if len(reqBody.Tools) == 0 {
		status, headers, respBody, err := proxyUpstreamRaw(body)
		if err != nil {
			logFooter(time.Since(start), "error")
			sendJSON(w, map[string]string{"error": "Upstream request failed: " + err.Error()}, http.StatusBadGateway)
			return
		}
		logResult("↪", fmt.Sprintf("Passthrough upstream response (%d)", status))
		logFooter(time.Since(start), "passthrough")

		for k, values := range headers {
			if strings.EqualFold(k, "Content-Length") {
				continue
			}
			for _, v := range values {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(status)
		w.Write(respBody)
		return
	}

	result := processChatRequest(reqBody)

	finishReason := result.Choices[0].FinishReason
	logFooter(time.Since(start), finishReason)
	sendJSON(w, result, http.StatusOK)
}

func handleDefault(w http.ResponseWriter, r *http.Request) {
	sendJSON(w, map[string]string{"status": "ok"}, http.StatusOK)
}

func sendJSON(w http.ResponseWriter, data interface{}, status int) {
	body, err := json.Marshal(data)
	if err != nil {
		http.Error(w, "JSON marshal error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}

func main() {
	initConfig()

	fmt.Println("Tool-calling proxy server")
	fmt.Printf("  Listening:  http://localhost:%s\n", ProxyPort)
	fmt.Printf("  Upstream:   %s\n", UpstreamURL)
	fmt.Printf("  Model:      %s\n", UpstreamModel)
	fmt.Println()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/", handleDefault)

	server := &http.Server{
		Addr:    ":" + ProxyPort,
		Handler: mux,
	}
	log.Fatal(server.ListenAndServe())
}
