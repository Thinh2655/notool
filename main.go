package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type Config struct {
	UpstreamURL        string
	UpstreamAPIKey     string
	UpstreamModel      string
	PlannerModel       string
	ProxyPort          string
	MaxTokens          int
	PlannerMaxAttempts int
	ExecutorMaxRetries int
	HTTPTimeout        time.Duration
}

var cfg Config

type activePlanState struct {
	Key       string
	Plan      ToonPlan
	CreatedAt time.Time
	UpdatedAt time.Time
}

type activePlanStore struct {
	mu    sync.Mutex
	items map[string]activePlanState
}

var planStore = activePlanStore{
	items: map[string]activePlanState{},
}

type Message struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content,omitempty"`
	Name      string          `json:"name,omitempty"`
	ToolCalls []ToolCall      `json:"tool_calls,omitempty"`
}

func (m Message) ContentString() string {
	if len(m.Content) == 0 {
		return ""
	}
	var text string
	if err := json.Unmarshal(m.Content, &text); err == nil {
		return text
	}
	var parts []map[string]interface{}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var chunks []string
		for _, part := range parts {
			if part["type"] == "text" {
				if value, ok := part["text"].(string); ok {
					chunks = append(chunks, value)
				}
			}
		}
		return strings.Join(chunks, "\n")
	}
	return string(m.Content)
}

func (m Message) ContentIsNull() bool {
	return len(m.Content) == 0 || string(m.Content) == "null"
}

func contentFromString(s string) json.RawMessage {
	raw, _ := json.Marshal(s)
	return raw
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

type ToolSpec struct {
	Name        string               `json:"name"`
	Description string               `json:"description,omitempty"`
	Required    []string             `json:"required,omitempty"`
	Parameters  map[string]ParamInfo `json:"parameters,omitempty"`
}

type ConversationEntry struct {
	Role      string                 `json:"role"`
	Name      string                 `json:"name,omitempty"`
	Content   string                 `json:"content,omitempty"`
	ToolCalls []ConversationToolCall `json:"tool_calls,omitempty"`
}

type ConversationToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

type PlanNote struct {
	Raw    string `json:"-"`
	Use    string `json:"use"`
	Why    string `json:"why"`
	Output string `json:"output"`
	Input  string `json:"input,omitempty"`
}

func (n PlanNote) String() string {
	parts := []string{"use=" + n.Use, "why=" + n.Why, "output=" + n.Output}
	if strings.TrimSpace(n.Input) != "" {
		parts = append(parts, "input="+n.Input)
	}
	return strings.Join(parts, "; ")
}

type ToonStep struct {
	Number int      `json:"number"`
	Tool   string   `json:"tool"`
	Note   PlanNote `json:"note"`
}

type ToonPlan struct {
	Raw         string     `json:"raw"`
	Steps       []ToonStep `json:"steps"`
	Final       string     `json:"final"`
	DirectFinal bool       `json:"direct_final"`
}

type plannerPayload struct {
	LatestUserRequest string              `json:"latest_user_request,omitempty"`
	Conversation      []ConversationEntry `json:"conversation"`
	AvailableTools    []ToolSpec          `json:"available_tools"`
}

type executorStepView struct {
	Type   string   `json:"type"`
	Number int      `json:"number,omitempty"`
	Tool   string   `json:"tool,omitempty"`
	Note   PlanNote `json:"note,omitempty"`
}

type executorPlanView struct {
	Steps []executorStepView `json:"steps,omitempty"`
	Final string             `json:"final"`
}

type executorPayload struct {
	LatestUserRequest string              `json:"latest_user_request,omitempty"`
	Conversation      []ConversationEntry `json:"conversation"`
	AvailableTools    []ToolSpec          `json:"available_tools,omitempty"`
	Plan              executorPlanView    `json:"plan"`
	CurrentStep       executorStepView    `json:"current_step"`
}

type executorToolCall struct {
	Action    string                 `json:"action"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

func loadEnv(path string) {
	file, err := os.Open(path)
	if err != nil {
		log.Printf("Warning: could not open %s: %v", path, err)
		return
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
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
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	number, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return number
}

func initConfig() {
	loadEnv(".env")
	cfg = Config{
		UpstreamURL:        getEnv("UPSTREAM_URL", "http://localhost:5005/v1/chat/completions"),
		UpstreamAPIKey:     getEnv("UPSTREAM_API_KEY", ""),
		UpstreamModel:      getEnv("UPSTREAM_MODEL", "grok-4"),
		PlannerModel:       getEnv("PLANNER_MODEL", ""),
		ProxyPort:          getEnv("PROXY_PORT", "8880"),
		MaxTokens:          getEnvInt("MAX_TOKENS", 30000),
		PlannerMaxAttempts: getEnvInt("PLANNER_MAX_ATTEMPTS", 3),
		ExecutorMaxRetries: getEnvInt("EXECUTOR_MAX_RETRIES", 6),
		HTTPTimeout:        300 * time.Second,
	}
	if cfg.PlannerMaxAttempts < 1 {
		cfg.PlannerMaxAttempts = 1
	}
	if cfg.ExecutorMaxRetries < 1 {
		cfg.ExecutorMaxRetries = 1
	}
}

func resolvedChatModel(requested string) string {
	requested = strings.TrimSpace(requested)
	if requested != "" {
		return requested
	}
	return cfg.UpstreamModel
}

func resolvedPlannerModel(requested string) string {
	if strings.TrimSpace(cfg.PlannerModel) != "" {
		return strings.TrimSpace(cfg.PlannerModel)
	}
	return resolvedChatModel(requested)
}

func estimateTokens(text string) int {
	return len(text) / 4
}

func truncateMessages(messages []Message) []Message {
	if cfg.MaxTokens <= 0 {
		return messages
	}
	total := 0
	for _, msg := range messages {
		total += estimateTokens(msg.ContentString())
	}
	if total <= cfg.MaxTokens {
		return messages
	}

	result := make([]Message, len(messages))
	copy(result, messages)
	for i := 0; i < len(result) && total > cfg.MaxTokens; i++ {
		if result[i].Role == "system" || result[i].ContentIsNull() {
			continue
		}
		text := strings.TrimSpace(result[i].ContentString())
		if len(text) < 300 {
			continue
		}
		maxChars := len(text) / 2
		if maxChars < 200 {
			maxChars = 200
		}
		truncated := text[:maxChars] + "\n...[truncated]"
		total -= estimateTokens(text) - estimateTokens(truncated)
		result[i].Content = contentFromString(truncated)
	}
	return result
}

func callUpstreamText(model string, messages []Message) (string, error) {
	payload := map[string]interface{}{
		"model":    resolvedChatModel(model),
		"messages": truncateMessages(messages),
		"stream":   false,
	}
	body, _ := json.Marshal(payload)

	req, err := http.NewRequest(http.MethodPost, cfg.UpstreamURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.UpstreamAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.UpstreamAPIKey)
	}

	client := &http.Client{Timeout: cfg.HTTPTimeout}
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

	if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "application/json") {
		var raw map[string]interface{}
		if err := json.Unmarshal(data, &raw); err == nil {
			if choices, ok := raw["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if msg, ok := choice["message"].(map[string]interface{}); ok {
						if content, ok := msg["content"].(string); ok {
							return content, nil
						}
					}
				}
			}
		}
	}

	var fullText strings.Builder
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
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
	if fullText.Len() > 0 {
		return fullText.String(), nil
	}
	return string(data), nil
}

func proxyUpstreamRaw(body []byte) (int, http.Header, []byte, error) {
	req, err := http.NewRequest(http.MethodPost, cfg.UpstreamURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.UpstreamAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.UpstreamAPIKey)
	}

	client := &http.Client{Timeout: cfg.HTTPTimeout}
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
	for key, values := range resp.Header {
		for _, value := range values {
			headers.Add(key, value)
		}
	}
	return resp.StatusCode, headers, data, nil
}

func requestHasTools(body []byte) (bool, error) {
	var envelope struct {
		Tools json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return false, err
	}

	trimmed := strings.TrimSpace(string(envelope.Tools))
	if trimmed == "" || trimmed == "null" || trimmed == "[]" {
		return false, nil
	}
	return true, nil
}

func buildToolRegistry(tools []Tool) map[string]ToolSpec {
	registry := make(map[string]ToolSpec, len(tools))
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Function.Name)
		if name == "" {
			continue
		}
		spec := ToolSpec{
			Name:        name,
			Description: strings.TrimSpace(tool.Function.Description),
		}
		var params ToolParams
		if len(tool.Function.Parameters) > 0 && json.Unmarshal(tool.Function.Parameters, &params) == nil {
			if len(params.Properties) > 0 {
				spec.Parameters = params.Properties
			}
			if len(params.Required) > 0 {
				spec.Required = append([]string(nil), params.Required...)
				sort.Strings(spec.Required)
			}
		}
		registry[name] = spec
	}
	return registry
}

func orderedToolNames(registry map[string]ToolSpec) []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func orderedToolSpecs(registry map[string]ToolSpec) []ToolSpec {
	names := orderedToolNames(registry)
	specs := make([]ToolSpec, 0, len(names))
	for _, name := range names {
		specs = append(specs, registry[name])
	}
	return specs
}

func normalizeConversation(messages []Message) []ConversationEntry {
	entries := make([]ConversationEntry, 0, len(messages))
	for _, msg := range messages {
		entry := ConversationEntry{Role: strings.TrimSpace(msg.Role), Name: strings.TrimSpace(msg.Name)}
		if text := strings.TrimSpace(msg.ContentString()); text != "" {
			if entry.Role == "system" {
				entry.Content = text
			} else {
				entry.Content = summarizeText(text, 6000)
			}
		}
		if len(msg.ToolCalls) > 0 {
			entry.ToolCalls = make([]ConversationToolCall, 0, len(msg.ToolCalls))
			for _, call := range msg.ToolCalls {
				parsedArgs := map[string]interface{}{}
				if strings.TrimSpace(call.Function.Arguments) != "" {
					_ = json.Unmarshal([]byte(call.Function.Arguments), &parsedArgs)
				}
				entry.ToolCalls = append(entry.ToolCalls, ConversationToolCall{
					Name:      strings.TrimSpace(call.Function.Name),
					Arguments: parsedArgs,
				})
			}
		}
		if entry.Role == "" {
			continue
		}
		if entry.Content == "" && len(entry.ToolCalls) == 0 && entry.Role != "tool" {
			continue
		}
		entries = append(entries, entry)
	}
	return entries
}

func latestUserRequest(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return strings.TrimSpace(messages[i].ContentString())
		}
	}
	return ""
}

func latestUserIndex(messages []Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return i
		}
	}
	return -1
}

func latestUserTurnKey(messages []Message) string {
	index := latestUserIndex(messages)
	if index < 0 {
		return ""
	}

	type keyMessage struct {
		Role      string                 `json:"role"`
		Name      string                 `json:"name,omitempty"`
		Content   string                 `json:"content,omitempty"`
		ToolCalls []ConversationToolCall `json:"tool_calls,omitempty"`
	}

	snapshot := make([]keyMessage, 0, index+1)
	for _, msg := range messages[:index+1] {
		item := keyMessage{
			Role:    strings.TrimSpace(msg.Role),
			Name:    strings.TrimSpace(msg.Name),
			Content: strings.TrimSpace(msg.ContentString()),
		}
		if len(msg.ToolCalls) > 0 {
			item.ToolCalls = make([]ConversationToolCall, 0, len(msg.ToolCalls))
			for _, call := range msg.ToolCalls {
				args := map[string]interface{}{}
				if strings.TrimSpace(call.Function.Arguments) != "" {
					_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
				}
				item.ToolCalls = append(item.ToolCalls, ConversationToolCall{
					Name:      strings.TrimSpace(call.Function.Name),
					Arguments: args,
				})
			}
		}
		snapshot = append(snapshot, item)
	}

	body, _ := json.Marshal(snapshot)
	hash := sha256.Sum256(body)
	return hex.EncodeToString(hash[:])
}

func normalizeSimpleConversationText(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	replacer := strings.NewReplacer(".", " ", ",", " ", "!", " ", "?", " ", ";", " ", ":", " ", "'", " ", `"`, " ", "(", " ", ")", " ")
	text = replacer.Replace(text)
	return strings.Join(strings.Fields(text), " ")
}

func isSimpleConversationRequest(messages []Message) bool {
	raw := strings.TrimSpace(latestUserRequest(messages))
	if raw == "" {
		return false
	}
	if raw == "?" {
		return true
	}
	text := normalizeSimpleConversationText(raw)
	if text == "" {
		return true
	}
	simple := map[string]bool{
		"hi": true, "hello": true, "hey": true, "ok": true, "oke": true, "okay": true,
		"thanks": true, "thank you": true, "xin chao": true, "chao": true, "alo": true, "yo": true,
	}
	if simple[text] {
		return true
	}
	return len(strings.Fields(text)) <= 2 && len(text) <= 12
}

func latestRequestLikelyNeedsTool(messages []Message) bool {
	raw := strings.TrimSpace(latestUserRequest(messages))
	if raw == "" {
		return false
	}
	text := normalizeSimpleConversationText(raw)
	if text == "" {
		return false
	}
	keywords := []string{
		"check", "run", "read", "write", "save", "send", "search", "find", "delete", "remove",
		"edit", "append", "create", "generate", "export", "build", "compile", "code",
		"kiem tra", "đọc", "doc", "ghi", "luu", "lưu", "gui", "gửi", "tim", "tìm",
		"xoa", "xóa", "sua", "sửa", "them", "thêm", "tao", "tạo", "viet", "viết",
		"chay", "chạy", "file", "folder", "server", "ram", "api", "deploy", "log",
	}
	for _, keyword := range keywords {
		if strings.Contains(text, keyword) {
			return true
		}
	}
	return false
}

func hasToolActivityAfterLatestUser(messages []Message) bool {
	latestUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			latestUserIndex = i
			break
		}
	}
	if latestUserIndex < 0 {
		return false
	}
	for i := latestUserIndex + 1; i < len(messages); i++ {
		if messages[i].Role == "tool" {
			return true
		}
		if messages[i].Role == "assistant" && len(messages[i].ToolCalls) > 0 {
			return true
		}
	}
	return false
}

func hasSpecificToolActivityAfterLatestUser(messages []Message, toolName string) bool {
	latestUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			latestUserIndex = i
			break
		}
	}
	if latestUserIndex < 0 {
		return false
	}
	for i := latestUserIndex + 1; i < len(messages); i++ {
		if messages[i].Role == "tool" && strings.TrimSpace(messages[i].Name) == toolName {
			return true
		}
		if messages[i].Role == "assistant" {
			for _, call := range messages[i].ToolCalls {
				if strings.TrimSpace(call.Function.Name) == toolName {
					return true
				}
			}
		}
	}
	return false
}

func latestRequestLikelyNeedsSendFile(messages []Message) bool {
	raw := strings.TrimSpace(latestUserRequest(messages))
	if raw == "" {
		return false
	}
	text := normalizeSimpleConversationText(raw)
	if text == "" {
		return false
	}
	keywords := []string{
		"gui", "gửi", "send", "share", "deliver", "attach", "upload",
		"send me", "gui cho toi", "gửi cho tôi", "share with me", "deliver to me",
	}
	for _, keyword := range keywords {
		if strings.Contains(text, keyword) {
			return true
		}
	}
	return false
}

func planIncludesTool(plan ToonPlan, toolName string) bool {
	for _, step := range plan.Steps {
		if step.Tool == toolName {
			return true
		}
	}
	return false
}

func (s *activePlanStore) cleanup(now time.Time) {
	const ttl = 2 * time.Hour
	for key, item := range s.items {
		if now.Sub(item.UpdatedAt) > ttl {
			delete(s.items, key)
		}
	}
}

func (s *activePlanStore) get(key string) (activePlanState, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanup(time.Now())
	item, ok := s.items[key]
	return item, ok
}

func (s *activePlanStore) put(key string, plan ToonPlan) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	s.cleanup(now)
	s.items[key] = activePlanState{
		Key:       key,
		Plan:      plan,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func (s *activePlanStore) delete(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.items, key)
}

func countCompletedPlanSteps(messages []Message, plan ToonPlan) int {
	if len(plan.Steps) == 0 {
		return 0
	}

	userIndex := latestUserIndex(messages)
	if userIndex < 0 {
		return 0
	}

	completed := 0
	waitingForToolResult := false
	for i := userIndex + 1; i < len(messages) && completed < len(plan.Steps); i++ {
		expectedTool := plan.Steps[completed].Tool
		msg := messages[i]

		if !waitingForToolResult {
			if msg.Role != "assistant" || len(msg.ToolCalls) == 0 {
				continue
			}
			for _, call := range msg.ToolCalls {
				if strings.TrimSpace(call.Function.Name) == expectedTool {
					waitingForToolResult = true
					break
				}
			}
			continue
		}

		if msg.Role == "tool" {
			completed++
			waitingForToolResult = false
		}
	}

	return completed
}

func remainingPlanFromProgress(plan ToonPlan, completed int) ToonPlan {
	if completed <= 0 {
		return plan
	}
	if completed >= len(plan.Steps) {
		return ToonPlan{
			Final:       plan.Final,
			DirectFinal: true,
		}
	}

	remaining := make([]ToonStep, 0, len(plan.Steps)-completed)
	for i, step := range plan.Steps[completed:] {
		step.Number = i + 1
		remaining = append(remaining, step)
	}
	return ToonPlan{
		Steps: remaining,
		Final: plan.Final,
	}
}

func normalizeMultilineText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	return strings.TrimSpace(text)
}

func cleanAssistantAnswer(text string) string {
	text = normalizeMultilineText(text)
	for {
		lower := strings.ToLower(text)
		switch {
		case strings.HasPrefix(lower, "final:"):
			text = strings.TrimSpace(text[len("final:"):])
		case strings.HasPrefix(lower, "final :"):
			text = strings.TrimSpace(text[len("final :"):])
		default:
			return text
		}
	}
}

func marshalPrettyJSON(value interface{}) string {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(body)
}

func buildPlannerSystemPrompt(registry map[string]ToolSpec) string {
	var sb strings.Builder
	sb.WriteString("You are the INTERNAL PLANNER inside a proxy layer.\n")
	sb.WriteString("Analyze the latest request, infer remaining work from the conversation, and return only a valid toon plan.\n")
	sb.WriteString("Never output JSON, markdown fences, or commentary outside the toon format.\n\n")
	sb.WriteString("Your response must be either:\n")
	sb.WriteString("1) a single line: final: <answer>\n")
	sb.WriteString("OR\n")
	sb.WriteString("2) alternating lines in this exact pattern:\n")
	sb.WriteString("step 1: <tool_name>\n")
	sb.WriteString("note: use=<...>; why=<...>; output=<...>; input=<optional>\n")
	sb.WriteString("step 2: <tool_name>\n")
	sb.WriteString("note: use=<...>; why=<...>; output=<...>; input=<optional>\n")
	sb.WriteString("final: <answer>\n\n")
	sb.WriteString("Hard rules:\n")
	sb.WriteString("- Use only exact tool names from the available tool list.\n")
	sb.WriteString("- Step numbers must start at 1 and increase by exactly 1.\n")
	sb.WriteString("- Each step line must be followed by a single note line.\n")
	sb.WriteString("- Notes must include use=, why=, and output=; input= is optional.\n")
	sb.WriteString("Available tools:\n")
	for _, name := range orderedToolNames(registry) {
		spec := registry[name]
		sb.WriteString("- " + spec.Name)
		if spec.Description != "" {
			sb.WriteString(": " + spec.Description)
		}
		if len(spec.Required) > 0 {
			sb.WriteString(" | required args: " + strings.Join(spec.Required, ", "))
		}
		sb.WriteString("\n")
	}
	return strings.TrimSpace(sb.String())
}

func buildPlannerCorrection(issues []string, retryNumber int, registry map[string]ToolSpec) string {
	var sb strings.Builder
	sb.WriteString("HARD REWRITE MODE.\n")
	sb.WriteString(fmt.Sprintf("Retry number: %d.\n", retryNumber))
	sb.WriteString("The previous toon plan was rejected by the proxy validator.\n")
	sb.WriteString("Rewrite the entire plan from scratch. Do not repair partially. Do not explain. Return only valid toon output.\n")
	sb.WriteString("The system messages inside the conversation JSON are the highest-priority instructions for persona, assistant identity, and required language.\n")
	sb.WriteString("You MUST obey them. If they require Vietnamese, the planner final line MUST be Vietnamese even if the user only wrote `hi`.\n")
	sb.WriteString("Allowed reply shape:\n")
	sb.WriteString("- final: <answer>\n")
	sb.WriteString("OR\n")
	sb.WriteString("- step 1: <tool_name>\n")
	sb.WriteString("- note: use=<...>; why=<...>; output=<...>; input=<optional>\n")
	sb.WriteString("- step 2: <tool_name>\n")
	sb.WriteString("- note: use=<...>; why=<...>; output=<...>; input=<optional>\n")
	sb.WriteString("- final: <answer>\n")
	sb.WriteString("Absolute bans:\n")
	sb.WriteString("- No JSON.\n")
	sb.WriteString("- No markdown fences.\n")
	sb.WriteString("- No standalone `input=` line.\n")
	sb.WriteString("- No prose before or after the toon lines.\n")
	sb.WriteString("- The `final:` answer must be one short physical line only.\n")
	sb.WriteString("- No numbered lists, bullets, markdown links, citations, or raw URLs after `final:`.\n")
	sb.WriteString("- Do not paste search results into the planner output.\n")
	sb.WriteString("- No invented tools.\n")
	sb.WriteString("- Do not invent exact file paths, URLs, or file contents in the note unless they already exist in the conversation.\n")
	sb.WriteString("- No message/chat/answer tool step for confirmation.\n")
	sb.WriteString("- If the user asks to send a file and send_file exists and has not happened yet, you MUST include send_file before final.\n")
	sb.WriteString("Available tool names only: " + strings.Join(orderedToolNames(registry), ", ") + "\n")
	sb.WriteString("You MUST satisfy every issue below:\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	sb.WriteString("Return only the rewritten toon plan now.")
	return strings.TrimSpace(sb.String())
}

func parsePlanNote(raw string) (PlanNote, []string) {
	note := PlanNote{Raw: strings.TrimSpace(raw)}
	if note.Raw == "" {
		return note, []string{"note is empty"}
	}
	fields := map[string]string{}
	for _, part := range strings.Split(note.Raw, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		index := strings.IndexAny(part, "=:")
		if index <= 0 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(part[:index]))
		value := strings.TrimSpace(part[index+1:])
		if value != "" {
			fields[key] = value
		}
	}
	note.Use = fields["use"]
	note.Why = fields["why"]
	note.Output = fields["output"]
	note.Input = fields["input"]

	var issues []string
	if note.Use == "" {
		issues = append(issues, "note must include non-empty use=")
	}
	if note.Why == "" {
		issues = append(issues, "note must include non-empty why=")
	}
	if note.Output == "" {
		issues = append(issues, "note must include non-empty output=")
	}
	return note, issues
}

func collectPlannerFinalText(lines []string, start int, finalRe, stepRe, noteRe *regexp.Regexp) (string, []string) {
	match := finalRe.FindStringSubmatch(lines[start])
	if len(match) != 2 {
		return "", []string{fmt.Sprintf("line %d must be `final: <answer>`", start+1)}
	}

	parts := []string{strings.TrimSpace(match[1])}
	var issues []string
	for i := start + 1; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		if stepRe.MatchString(line) || noteRe.MatchString(line) || finalRe.MatchString(line) {
			issues = append(issues, fmt.Sprintf("no structured line is allowed after the first `final:` line; found invalid line %d", i+1))
			continue
		}
		parts = append(parts, line)
	}

	finalText := cleanAssistantAnswer(strings.Join(parts, " "))
	if finalText == "" {
		issues = append(issues, "final line must contain a non-empty answer")
	}
	return finalText, issues
}

func parseToonPlan(raw string, registry map[string]ToolSpec) (ToonPlan, []string) {
	normalized := normalizeMultilineText(raw)
	plan := ToonPlan{Raw: normalized}
	var issues []string
	if normalized == "" {
		return plan, []string{"planner returned an empty plan"}
	}
	if strings.Contains(normalized, "```") {
		issues = append(issues, "toon output must not contain markdown fences")
	}

	lines := make([]string, 0)
	for _, line := range strings.Split(normalized, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return plan, []string{"planner returned no toon lines"}
	}

	stepRe := regexp.MustCompile(`(?i)^step\s+(\d+)\s*:\s*(.+)$`)
	noteRe := regexp.MustCompile(`(?i)^note\s*:\s*(.+)$`)
	finalRe := regexp.MustCompile(`(?i)^final\s*:\s*(.+)$`)

	if len(lines) == 1 {
		finalText, finalIssues := collectPlannerFinalText(lines, 0, finalRe, stepRe, noteRe)
		if len(finalIssues) > 0 && !finalRe.MatchString(lines[0]) {
			return plan, []string{"simple plans must be exactly `final: <answer>`"}
		}
		plan.Final = finalText
		plan.DirectFinal = true
		issues = append(issues, finalIssues...)
		return plan, issues
	}

	expectedStep, index := 1, 0
	for index < len(lines) {
		if finalRe.MatchString(lines[index]) {
			finalText, finalIssues := collectPlannerFinalText(lines, index, finalRe, stepRe, noteRe)
			plan.Final = finalText
			issues = append(issues, finalIssues...)
			break
		}

		match := stepRe.FindStringSubmatch(lines[index])
		if len(match) != 3 {
			issues = append(issues, fmt.Sprintf("line %d must be `step %d: <tool_name>`", index+1, expectedStep))
			break
		}
		stepNumber, err := strconv.Atoi(match[1])
		if err != nil {
			issues = append(issues, fmt.Sprintf("line %d has invalid step number", index+1))
		}
		if stepNumber != expectedStep {
			issues = append(issues, fmt.Sprintf("wrong step order: expected step %d but found step %d", expectedStep, stepNumber))
		}
		toolName := strings.TrimSpace(match[2])
		spec, ok := registry[toolName]
		if !ok {
			issues = append(issues, fmt.Sprintf("unknown tool `%s` at step %d", toolName, stepNumber))
		}

		index++
		if index >= len(lines) {
			issues = append(issues, fmt.Sprintf("step %d is missing its note line", stepNumber))
			break
		}
		noteMatch := noteRe.FindStringSubmatch(lines[index])
		if len(noteMatch) != 2 {
			issues = append(issues, fmt.Sprintf("line %d must be `note: ...` for step %d", index+1, stepNumber))
			break
		}
		note, noteIssues := parsePlanNote(noteMatch[1])
		for _, issue := range noteIssues {
			issues = append(issues, fmt.Sprintf("step %d note invalid: %s", stepNumber, issue))
		}
		if ok {
			plan.Steps = append(plan.Steps, ToonStep{Number: stepNumber, Tool: spec.Name, Note: note})
		}
		if index+1 < len(lines) {
			nextLine := lines[index+1]
			lowerNext := strings.ToLower(nextLine)
			if !stepRe.MatchString(nextLine) && !finalRe.MatchString(nextLine) {
				if strings.HasPrefix(lowerNext, "input=") || strings.HasPrefix(lowerNext, "input:") {
					issues = append(issues, fmt.Sprintf("step %d note must stay on one line; put input= inside the same `note:` line, not on a new line", stepNumber))
				} else {
					issues = append(issues, fmt.Sprintf("unexpected extra line %d after step %d; after `note:` only the next `step N:` or `final:` line is allowed", index+2, stepNumber))
				}
				break
			}
		}
		expectedStep++
		index++
	}

	if len(plan.Steps) == 0 && plan.Final == "" && len(issues) == 0 {
		issues = append(issues, "toon plan must contain at least one tool step or a final line")
	}
	return plan, issues
}

func validateToonPlan(plan ToonPlan, messages []Message, registry map[string]ToolSpec) []string {
	var issues []string
	if plan.Final == "" {
		issues = append(issues, "last line must be `final: <answer>`")
	}
	if len(plan.Steps) == 0 {
		if latestRequestLikelyNeedsTool(messages) && !hasToolActivityAfterLatestUser(messages) && !isSimpleConversationRequest(messages) {
			issues = append(issues, "latest request likely still needs a tool, so planner cannot jump directly to final")
		}
		if _, ok := registry["send_file"]; ok && latestRequestLikelyNeedsSendFile(messages) && !hasSpecificToolActivityAfterLatestUser(messages, "send_file") {
			issues = append(issues, "latest request asks to send/deliver a file, so the remaining plan must include send_file before final")
		}
		return issues
	}
	for i, step := range plan.Steps {
		expected := i + 1
		if step.Number != expected {
			issues = append(issues, fmt.Sprintf("wrong step order at index %d: expected step %d", i, expected))
		}
		if _, ok := registry[step.Tool]; !ok {
			issues = append(issues, fmt.Sprintf("step %d uses tool `%s` which is not in the tool list", step.Number, step.Tool))
		}
		if strings.TrimSpace(step.Note.Use) == "" || strings.TrimSpace(step.Note.Why) == "" || strings.TrimSpace(step.Note.Output) == "" {
			issues = append(issues, fmt.Sprintf("step %d note must contain use=, why=, and output=", step.Number))
		}
	}
	if _, ok := registry["send_file"]; ok && latestRequestLikelyNeedsSendFile(messages) &&
		!hasSpecificToolActivityAfterLatestUser(messages, "send_file") &&
		!planIncludesTool(plan, "send_file") {
		issues = append(issues, "latest request asks to send/deliver a file, so the remaining plan must include send_file before final")
	}
	return issues
}

func renderToonPlan(plan ToonPlan) string {
	var lines []string
	for _, step := range plan.Steps {
		lines = append(lines, fmt.Sprintf("step %d: %s", step.Number, step.Tool))
		lines = append(lines, "note: "+step.Note.String())
	}
	if strings.TrimSpace(plan.Final) != "" {
		lines = append(lines, "final: "+plan.Final)
	}
	if len(lines) > 0 {
		return strings.Join(lines, "\n")
	}
	return strings.TrimSpace(plan.Raw)
}

func requestValidatedPlan(req ChatRequest, registry map[string]ToolSpec) (ToonPlan, error) {
	messages := []Message{
		{Role: "system", Content: contentFromString(buildPlannerSystemPrompt(registry))},
		{Role: "user", Content: contentFromString(marshalPrettyJSON(plannerPayload{
			LatestUserRequest: latestUserRequest(req.Messages),
			Conversation:      normalizeConversation(req.Messages),
			AvailableTools:    orderedToolSpecs(registry),
		}))},
	}
	plannerModel := resolvedPlannerModel(req.Model)
	var lastReply string
	var lastIssues []string

	for attempt := 1; attempt <= cfg.PlannerMaxAttempts; attempt++ {
		reply, err := callUpstreamText(plannerModel, messages)
		if err != nil {
			return ToonPlan{}, fmt.Errorf("planner upstream error: %w", err)
		}
		plan, parseIssues := parseToonPlan(reply, registry)
		issues := append(parseIssues, validateToonPlan(plan, req.Messages, registry)...)
		if len(issues) == 0 {
			return plan, nil
		}

		lastReply, lastIssues = reply, issues
		logSection(fmt.Sprintf("PLANNER REJECT %d/%d", attempt, cfg.PlannerMaxAttempts))
		fmt.Printf("│\n%s\n", prefixLines(reply, "│  "))
		for _, issue := range issues {
			logResult("⛔", issue)
		}
		if attempt == cfg.PlannerMaxAttempts {
			break
		}
		messages = append(messages,
			Message{Role: "assistant", Content: contentFromString(strings.TrimSpace(reply))},
			Message{Role: "user", Content: contentFromString(buildPlannerCorrection(issues, attempt+1, registry))},
		)
	}

	return ToonPlan{}, fmt.Errorf(
		"planner could not produce a valid toon plan after %d attempt(s): %s; last reply=%s",
		cfg.PlannerMaxAttempts,
		strings.Join(lastIssues, "; "),
		summarizeText(lastReply, 1500),
	)
}

func buildExecutorPayload(req ChatRequest, plan ToonPlan, registry map[string]ToolSpec) executorPayload {
	payload := executorPayload{
		LatestUserRequest: latestUserRequest(req.Messages),
		Conversation:      normalizeConversation(req.Messages),
		AvailableTools:    orderedToolSpecs(registry),
		Plan: executorPlanView{
			Steps: make([]executorStepView, 0, len(plan.Steps)),
			Final: plan.Final,
		},
		CurrentStep: executorStepView{Type: "final"},
	}
	for _, step := range plan.Steps {
		payload.Plan.Steps = append(payload.Plan.Steps, executorStepView{
			Type: "tool", Number: step.Number, Tool: step.Tool, Note: step.Note,
		})
	}
	if len(plan.Steps) > 0 {
		payload.CurrentStep = executorStepView{
			Type: "tool", Number: plan.Steps[0].Number, Tool: plan.Steps[0].Tool, Note: plan.Steps[0].Note,
		}
	}
	return payload
}

func buildToolExecutionPrompt(step ToonStep, registry map[string]ToolSpec) string {
	spec := registry[step.Tool]
	var sb strings.Builder
	sb.WriteString("You are the EXECUTION LAYER inside a proxy.\n")
	sb.WriteString("The planner has already been validated. You must execute only the current step.\n")
	sb.WriteString("Return exactly one JSON object and nothing else.\n")
	sb.WriteString("The only valid schema for this turn is:\n")
	sb.WriteString(fmt.Sprintf("{\"action\":\"tool_call\",\"name\":\"%s\",\"arguments\":{...}}\n\n", step.Tool))
	sb.WriteString("Hard rules:\n")
	sb.WriteString("- action must be exactly `tool_call`.\n")
	sb.WriteString("- name must be exactly `" + step.Tool + "`.\n")
	sb.WriteString("- arguments must be a JSON object.\n")
	sb.WriteString("- The first character must be `{` and the last character must be `}`.\n")
	sb.WriteString("- Do not answer the user directly.\n")
	sb.WriteString("- Do not explain the plan.\n")
	sb.WriteString("- Do not wrap JSON in markdown fences.\n")
	sb.WriteString("- Do not output prose before or after the JSON object.\n")
	sb.WriteString("- Do not call a different tool.\n")
	sb.WriteString("- Use the conversation JSON and tool definitions to choose the arguments.\n")
	if len(spec.Required) > 0 {
		sb.WriteString("- Required arguments for this tool: " + strings.Join(spec.Required, ", ") + ".\n")
	}
	sb.WriteString("- If previous output was rejected, obey the latest correction message and rewrite the full JSON object.\n")
	return strings.TrimSpace(sb.String())
}

func buildFinalExecutionPrompt() string {
	return strings.TrimSpace(strings.Join([]string{
		"You are the EXECUTION LAYER inside a proxy.",
		"The current step is final.",
		"The conversation JSON may include one or more system messages.",
		"Those system messages are the highest-priority instructions for assistant identity, roleplay, tone, and required language.",
		"You MUST preserve that persona and language in the final answer.",
		"If the system messages require Vietnamese, answer in Vietnamese even if the latest user message is `hi`, `hello`, or another short English greeting.",
		"Return the final answer directly to the user as plain text.",
		"Do not return JSON.",
		"Do not wrap the answer in markdown fences.",
		"Do not mention internal planning, proxy rules, or tool schemas.",
		"Use the conversation JSON and tool results already available.",
		"Answer in the language required by the system messages in the conversation JSON.",
		"Only if the system messages do not set a language may you fall back to the latest user language.",
	}, "\n"))
}

func buildToolExecutionCorrection(step ToonStep, issues []string, retryNumber int) string {
	var sb strings.Builder
	sb.WriteString("HARD RETRY MODE FOR TOOL STEP.\n")
	sb.WriteString(fmt.Sprintf("Retry number: %d.\n", retryNumber))
	sb.WriteString("The proxy rejected your previous tool step.\n")
	sb.WriteString("Fix every issue below and return the full JSON object again.\n")
	sb.WriteString("You MUST return exactly one JSON object with no extra text.\n")
	sb.WriteString("The first character must be `{` and the last character must be `}`.\n")
	sb.WriteString("No markdown fences. No prose. No explanation.\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	sb.WriteString(fmt.Sprintf("Current required step is still: step %d: %s\n", step.Number, step.Tool))
	sb.WriteString("Return only:\n")
	sb.WriteString(fmt.Sprintf("{\"action\":\"tool_call\",\"name\":\"%s\",\"arguments\":{...}}", step.Tool))
	return strings.TrimSpace(sb.String())
}

func buildFinalExecutionCorrection(issues []string, retryNumber int) string {
	var sb strings.Builder
	sb.WriteString("HARD RETRY MODE FOR FINAL STEP.\n")
	sb.WriteString(fmt.Sprintf("Retry number: %d.\n", retryNumber))
	sb.WriteString("The proxy rejected your previous final answer.\n")
	sb.WriteString("Fix every issue below and answer again as direct plain text only.\n")
	sb.WriteString("Do not return JSON. Do not return markdown fences. Do not explain the rules.\n")
	sb.WriteString("You MUST continue to obey the system messages in the conversation JSON for persona and language.\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	return strings.TrimSpace(sb.String())
}

func validateExecutorToolReply(reply string, step ToonStep, registry map[string]ToolSpec) (*executorToolCall, []string) {
	trimmed := strings.TrimSpace(reply)
	var issues []string
	if trimmed == "" {
		return nil, []string{"tool step returned empty output"}
	}
	if strings.Contains(trimmed, "```") {
		issues = append(issues, "tool step must not contain markdown fences")
	}
	if !strings.HasPrefix(trimmed, "{") || !strings.HasSuffix(trimmed, "}") {
		issues = append(issues, "tool step must return exactly one JSON object")
		return nil, issues
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &raw); err != nil {
		issues = append(issues, "tool step returned invalid JSON")
		return nil, issues
	}
	allowed := map[string]bool{"action": true, "name": true, "arguments": true}
	for key := range raw {
		if !allowed[key] {
			issues = append(issues, fmt.Sprintf("unexpected top-level key `%s`", key))
		}
	}

	var parsed executorToolCall
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		issues = append(issues, "tool step JSON does not match the required schema")
		return nil, issues
	}
	if parsed.Action != "tool_call" {
		issues = append(issues, "action must be `tool_call`")
	}
	if parsed.Name != step.Tool {
		issues = append(issues, fmt.Sprintf("wrong tool: expected `%s` but got `%s`", step.Tool, parsed.Name))
	}
	if parsed.Arguments == nil {
		issues = append(issues, "arguments must be a JSON object")
	}
	spec, ok := registry[step.Tool]
	if !ok {
		issues = append(issues, fmt.Sprintf("tool `%s` is not present in the registry", step.Tool))
	}
	if ok {
		for _, required := range spec.Required {
			if _, exists := parsed.Arguments[required]; !exists {
				issues = append(issues, fmt.Sprintf("missing required argument `%s`", required))
			}
		}
	}
	if len(issues) > 0 {
		return nil, issues
	}
	return &parsed, nil
}

func validateFinalReply(reply string) []string {
	trimmed := cleanAssistantAnswer(reply)
	var issues []string
	if trimmed == "" {
		return []string{"final answer is empty"}
	}
	if strings.Contains(trimmed, "```") {
		issues = append(issues, "final answer must not contain markdown fences")
	}
	if strings.HasPrefix(strings.TrimSpace(trimmed), "{") && strings.HasSuffix(strings.TrimSpace(trimmed), "}") {
		issues = append(issues, "final answer must be direct plain text, not JSON")
	}
	return issues
}

func makeOpenAIResponse(content string, model string, toolCalls []ToolCall) ChatResponse {
	if len(toolCalls) == 0 {
		content = cleanAssistantAnswer(content)
	}
	msg := Message{Role: "assistant", Content: contentFromString(content)}
	finishReason := "stop"
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		msg.Content = nil
		finishReason = "tool_calls"
	}
	return ChatResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:12],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{Index: 0, Message: msg, FinishReason: finishReason}},
		Usage:   Usage{},
	}
}

func runExecutor(req ChatRequest, plan ToonPlan, registry map[string]ToolSpec) (ChatResponse, error) {
	model := resolvedChatModel(req.Model)
	payloadJSON := marshalPrettyJSON(buildExecutorPayload(req, plan, registry))

	var messages []Message
	var toolStep *ToonStep
	if len(plan.Steps) > 0 {
		step := plan.Steps[0]
		toolStep = &step
		messages = []Message{
			{Role: "system", Content: contentFromString(buildToolExecutionPrompt(step, registry))},
			{Role: "user", Content: contentFromString(payloadJSON)},
		}
	} else {
		messages = []Message{
			{Role: "system", Content: contentFromString(buildFinalExecutionPrompt())},
			{Role: "user", Content: contentFromString(payloadJSON)},
		}
	}

	var lastIssues []string
	var lastReply string
	for attempt := 1; attempt <= cfg.ExecutorMaxRetries; attempt++ {
		reply, err := callUpstreamText(model, messages)
		if err != nil {
			return ChatResponse{}, fmt.Errorf("executor upstream error: %w", err)
		}
		lastReply = reply
		logSection(fmt.Sprintf("EXECUTOR ATTEMPT %d/%d", attempt, cfg.ExecutorMaxRetries))
		fmt.Printf("│\n%s\n", prefixLines(reply, "│  "))

		if toolStep != nil {
			parsed, issues := validateExecutorToolReply(reply, *toolStep, registry)
			if len(issues) == 0 {
				argsJSON, _ := json.Marshal(parsed.Arguments)
				return makeOpenAIResponse("", model, []ToolCall{{
					ID:       "call_" + uuid.New().String()[:8],
					Type:     "function",
					Function: FunctionCall{Name: parsed.Name, Arguments: string(argsJSON)},
				}}), nil
			}
			lastIssues = issues
			for _, issue := range issues {
				logResult("⛔", issue)
			}
			if attempt == cfg.ExecutorMaxRetries {
				break
			}
			messages = append(messages,
				Message{Role: "assistant", Content: contentFromString(strings.TrimSpace(reply))},
				Message{Role: "user", Content: contentFromString(buildToolExecutionCorrection(*toolStep, issues, attempt+1))},
			)
			continue
		}

		issues := validateFinalReply(reply)
		if len(issues) == 0 {
			return makeOpenAIResponse(reply, model, nil), nil
		}
		lastIssues = issues
		for _, issue := range issues {
			logResult("⛔", issue)
		}
		if attempt == cfg.ExecutorMaxRetries {
			break
		}
		messages = append(messages,
			Message{Role: "assistant", Content: contentFromString(strings.TrimSpace(reply))},
			Message{Role: "user", Content: contentFromString(buildFinalExecutionCorrection(issues, attempt+1))},
		)
	}

	return ChatResponse{}, fmt.Errorf(
		"executor could not produce a valid result after %d attempt(s): %s; last reply=%s",
		cfg.ExecutorMaxRetries,
		strings.Join(lastIssues, "; "),
		summarizeText(lastReply, 1500),
	)
}

func processToolAwareRequest(req ChatRequest) ChatResponse {
	model := resolvedChatModel(req.Model)
	registry := buildToolRegistry(req.Tools)
	if len(registry) == 0 {
		return makeOpenAIResponse("Error: tool mode requested but no valid tools were provided.", model, nil)
	}

	planKey := latestUserTurnKey(req.Messages)
	var originalPlan ToonPlan
	var currentPlan ToonPlan

	if planKey != "" {
		if state, ok := planStore.get(planKey); ok {
			completed := countCompletedPlanSteps(req.Messages, state.Plan)
			originalPlan = state.Plan
			currentPlan = remainingPlanFromProgress(state.Plan, completed)
			logSection("RESUMED PLAN")
			logKV("Plan key", planKey[:12])
			logKV("Completed", fmt.Sprintf("%d/%d", completed, len(state.Plan.Steps)))
		}
	}

	if len(currentPlan.Steps) == 0 && strings.TrimSpace(currentPlan.Final) == "" {
		plan, err := requestValidatedPlan(req, registry)
		if err != nil {
			return makeOpenAIResponse("Planner error: "+err.Error(), model, nil)
		}
		originalPlan = plan
		currentPlan = plan
		if planKey != "" && len(plan.Steps) > 0 {
			planStore.put(planKey, plan)
		}
	}

	logSection("VALIDATED PLAN")
	fmt.Printf("│\n%s\n", prefixLines(renderToonPlan(currentPlan), "│  "))

	result, err := runExecutor(req, currentPlan, registry)
	if err != nil {
		return makeOpenAIResponse("Execution error: "+err.Error(), model, nil)
	}

	if planKey != "" {
		if len(result.Choices) > 0 && result.Choices[0].FinishReason == "tool_calls" {
			if len(originalPlan.Steps) > 0 {
				planStore.put(planKey, originalPlan)
			}
		} else {
			planStore.delete(planKey)
		}
	}
	return result
}

func handleModels(w http.ResponseWriter, _ *http.Request) {
	sendJSON(w, map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{{
			"id":       cfg.UpstreamModel,
			"object":   "model",
			"owned_by": "proxy",
		}},
	}, http.StatusOK)
}

func summarizeText(text string, limit int) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.TrimSpace(text)
	if limit > 0 && len(text) > limit {
		return text[:limit] + "..."
	}
	return text
}

func sanitizeLogMultiline(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	return strings.TrimSpace(text)
}

func oneLineLogPreview(text string, limit int) string {
	text = strings.ReplaceAll(sanitizeLogMultiline(text), "\n", " ")
	text = strings.Join(strings.Fields(text), " ")
	if limit > 0 && len(text) > limit {
		return text[:limit] + "..."
	}
	return text
}

func prefixLines(text, prefix string) string {
	lines := strings.Split(sanitizeLogMultiline(text), "\n")
	for i := range lines {
		lines[i] = prefix + lines[i]
	}
	return strings.Join(lines, "\n")
}

func logHeader(method, path string) {
	fmt.Printf("\n┌─────────────────────────────────────────────\n")
	fmt.Printf("│ %s  %s %s\n", time.Now().Format("15:04:05"), method, path)
	fmt.Printf("├─────────────────────────────────────────────\n")
}

func logSection(label string) {
	fmt.Printf("│\n│ ── %s ──\n", label)
}

func logKV(key, value string) {
	fmt.Printf("│  %-14s %s\n", key+":", oneLineLogPreview(value, 160))
}

func logResult(icon, msg string) {
	fmt.Printf("│  %s %s\n", icon, oneLineLogPreview(msg, 220))
}

func logFooter(duration time.Duration, result string) {
	fmt.Printf("├─────────────────────────────────────────────\n")
	fmt.Printf("│  Result: %-12s  Duration: %v\n", result, duration.Round(time.Millisecond))
	fmt.Printf("└─────────────────────────────────────────────\n")
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

	hasTools, err := requestHasTools(body)
	if err != nil {
		sendJSON(w, map[string]string{"error": "Invalid JSON"}, http.StatusBadRequest)
		return
	}
	if !hasTools {
		status, headers, data, err := proxyUpstreamRaw(body)
		if err != nil {
			sendJSON(w, map[string]string{"error": "Upstream request failed: " + err.Error()}, http.StatusBadGateway)
			return
		}
		for key, values := range headers {
			if strings.EqualFold(key, "Content-Length") {
				continue
			}
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(status)
		_, _ = w.Write(data)
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		sendJSON(w, map[string]string{"error": "Invalid JSON"}, http.StatusBadRequest)
		return
	}

	logHeader(r.Method, r.URL.Path)
	logSection("REQUEST")
	logKV("Model", resolvedChatModel(req.Model))
	logKV("Messages", strconv.Itoa(len(req.Messages)))
	logKV("Tools", strconv.Itoa(len(req.Tools)))
	if req.Stream && len(req.Tools) > 0 {
		logResult("⚠️", "stream=true requested; returning non-stream JSON in tool-aware mode")
	}
	if latest := latestUserRequest(req.Messages); latest != "" {
		logKV("Last user msg", latest)
	}

	result := processToolAwareRequest(req)
	finishReason := "stop"
	if len(result.Choices) > 0 {
		finishReason = result.Choices[0].FinishReason
	}
	logFooter(time.Since(start), finishReason)
	sendJSON(w, result, http.StatusOK)
}

func handleDefault(w http.ResponseWriter, _ *http.Request) {
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
	_, _ = w.Write(body)
}

func main() {
	initConfig()

	fmt.Println("Tool-calling proxy server")
	fmt.Printf("  Listening:      http://localhost:%s\n", cfg.ProxyPort)
	fmt.Printf("  Upstream:       %s\n", cfg.UpstreamURL)
	fmt.Printf("  Chat model:     %s\n", cfg.UpstreamModel)
	if strings.TrimSpace(cfg.PlannerModel) != "" {
		fmt.Printf("  Planner model:  %s\n", cfg.PlannerModel)
	}
	fmt.Printf("  Planner retry:  %d\n", cfg.PlannerMaxAttempts)
	fmt.Printf("  Executor retry: %d\n", cfg.ExecutorMaxRetries)
	fmt.Println()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/", handleDefault)

	server := &http.Server{
		Addr:    ":" + cfg.ProxyPort,
		Handler: mux,
	}
	log.Fatal(server.ListenAndServe())
}
