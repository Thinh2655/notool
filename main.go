package main

import (
	"bufio"
	"bytes"
	"context"
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
	toon "github.com/toon-format/toon-go"
)

type Config struct {
	UpstreamURL        string
	UpstreamAPIKey     string
	UpstreamModel      string
	PlannerModel       string
	ProxyPort          string
	MaxTokens          int
	MaxRequestBytes    int
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
	LatestUserRequest string                `json:"latest_user_request,omitempty"`
	Conversation      []ConversationEntry   `json:"conversation"`
	AvailableTools    []ToolSpec            `json:"available_tools"`
	ExecutionTrace    []executionTraceEntry `json:"execution_trace,omitempty"`
}

type executionTraceEntry struct {
	Kind    string `json:"kind"`
	Tool    string `json:"tool,omitempty"`
	Summary string `json:"summary"`
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

type toonConversationToolCallDoc struct {
	Name      string            `toon:"name"`
	Arguments map[string]string `toon:"arguments,omitempty"`
}

type toonConversationEntryDoc struct {
	Role      string                        `toon:"role"`
	Name      string                        `toon:"name,omitempty"`
	Content   string                        `toon:"content,omitempty"`
	ToolCalls []toonConversationToolCallDoc `toon:"tool_calls,omitempty"`
}

type toonParameterDoc struct {
	Name        string `toon:"name"`
	Type        string `toon:"type"`
	Description string `toon:"description,omitempty"`
}

type toonToolSpecDoc struct {
	Name        string             `toon:"name"`
	Description string             `toon:"description,omitempty"`
	Required    []string           `toon:"required,omitempty"`
	Parameters  []toonParameterDoc `toon:"parameters,omitempty"`
}

type toonExecutionTraceDoc struct {
	Kind    string `toon:"kind"`
	Tool    string `toon:"tool,omitempty"`
	Summary string `toon:"summary"`
}

type toonPlanNoteDoc struct {
	Use    string `toon:"use,omitempty"`
	Why    string `toon:"why,omitempty"`
	Output string `toon:"output,omitempty"`
	Input  string `toon:"input,omitempty"`
}

type toonExecutorStepDoc struct {
	Type   string           `toon:"type"`
	Number int              `toon:"number,omitempty"`
	Tool   string           `toon:"tool,omitempty"`
	Note   *toonPlanNoteDoc `toon:"note,omitempty"`
}

type toonExecutorPlanDoc struct {
	Steps []toonExecutorStepDoc `toon:"steps,omitempty"`
	Final string                `toon:"final,omitempty"`
}

type toonPlannerPayloadDoc struct {
	LatestUserRequest string                     `toon:"LATEST_USER_REQUEST,omitempty"`
	Conversation      []toonConversationEntryDoc `toon:"CONVERSATION"`
	AvailableTools    []toonToolSpecDoc          `toon:"AVAILABLE_TOOLS,omitempty"`
	ExecutionTrace    []toonExecutionTraceDoc    `toon:"EXECUTION_TRACE,omitempty"`
}

type toonExecutorPayloadDoc struct {
	LatestUserRequest string                     `toon:"LATEST_USER_REQUEST,omitempty"`
	Conversation      []toonConversationEntryDoc `toon:"CONVERSATION"`
	AvailableTools    []toonToolSpecDoc          `toon:"AVAILABLE_TOOLS,omitempty"`
	Plan              toonExecutorPlanDoc        `toon:"PLAN"`
	CurrentStep       toonExecutorStepDoc        `toon:"CURRENT_STEP"`
}

type compactionProfile struct {
	TargetTokens    int
	TargetBytes     int
	RecentMessages  int
	SystemLimit     int
	LatestUserLimit int
	UserLimit       int
	AssistantLimit  int
	ToolLimit       int
	ToolArgLimit    int
	PreviewLimit    int
	SummaryLines    int
	SummaryLimit    int
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
		MaxRequestBytes:    getEnvInt("MAX_REQUEST_BYTES", 180000),
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

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func compactText(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	return summarizeText(normalizeMultilineText(text), limit)
}

func compactInlineText(text string, limit int) string {
	text = strings.Join(strings.Fields(normalizeMultilineText(text)), " ")
	return compactText(text, limit)
}

func compactToolCallArguments(raw string, limit int) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	return compactInlineText(formatToolCallArguments(raw), limit)
}

func compactToolCalls(calls []ToolCall, limit int) []ToolCall {
	if len(calls) == 0 {
		return nil
	}
	result := make([]ToolCall, 0, len(calls))
	for _, call := range calls {
		result = append(result, ToolCall{
			ID:   strings.TrimSpace(call.ID),
			Type: strings.TrimSpace(call.Type),
			Function: FunctionCall{
				Name:      strings.TrimSpace(call.Function.Name),
				Arguments: compactToolCallArguments(call.Function.Arguments, limit),
			},
		})
	}
	return result
}

func estimateMessageTokens(messages []Message, toolArgLimit int) int {
	total := 0
	for _, msg := range messages {
		total += estimateTokens(msg.ContentString())
		for _, call := range msg.ToolCalls {
			total += estimateTokens(strings.TrimSpace(call.Function.Name))
			total += estimateTokens(compactToolCallArguments(call.Function.Arguments, toolArgLimit))
		}
	}
	return total
}

func estimateMessageBytes(messages []Message) int {
	body, err := json.Marshal(messages)
	if err != nil {
		return 0
	}
	return len(body)
}

func compactionProfileForAttempt(attempt int) compactionProfile {
	switch attempt {
	case 0:
		return compactionProfile{
			TargetTokens:    cfg.MaxTokens,
			TargetBytes:     cfg.MaxRequestBytes,
			RecentMessages:  18,
			SystemLimit:     8000,
			LatestUserLimit: 6000,
			UserLimit:       3000,
			AssistantLimit:  2400,
			ToolLimit:       1800,
			ToolArgLimit:    900,
			PreviewLimit:    220,
			SummaryLines:    8,
			SummaryLimit:    2200,
		}
	case 1:
		return compactionProfile{
			TargetTokens:    maxInt(6000, cfg.MaxTokens*3/4),
			TargetBytes:     maxInt(64000, cfg.MaxRequestBytes*3/4),
			RecentMessages:  12,
			SystemLimit:     4500,
			LatestUserLimit: 4000,
			UserLimit:       1800,
			AssistantLimit:  1400,
			ToolLimit:       900,
			ToolArgLimit:    500,
			PreviewLimit:    180,
			SummaryLines:    6,
			SummaryLimit:    1600,
		}
	case 2:
		return compactionProfile{
			TargetTokens:    maxInt(4000, cfg.MaxTokens/2),
			TargetBytes:     maxInt(32000, cfg.MaxRequestBytes/2),
			RecentMessages:  8,
			SystemLimit:     2600,
			LatestUserLimit: 2400,
			UserLimit:       1000,
			AssistantLimit:  750,
			ToolLimit:       500,
			ToolArgLimit:    320,
			PreviewLimit:    140,
			SummaryLines:    5,
			SummaryLimit:    1100,
		}
	default:
		return compactionProfile{
			TargetTokens:    maxInt(2500, cfg.MaxTokens/3),
			TargetBytes:     maxInt(20000, cfg.MaxRequestBytes/3),
			RecentMessages:  5,
			SystemLimit:     1800,
			LatestUserLimit: 1600,
			UserLimit:       520,
			AssistantLimit:  420,
			ToolLimit:       280,
			ToolArgLimit:    180,
			PreviewLimit:    110,
			SummaryLines:    4,
			SummaryLimit:    750,
		}
	}
}

func messageContentLimit(msg Message, index, latestUserIdx int, profile compactionProfile) int {
	role := strings.TrimSpace(msg.Role)
	switch role {
	case "system":
		return profile.SystemLimit
	case "tool":
		return profile.ToolLimit
	case "user":
		if index == latestUserIdx {
			return profile.LatestUserLimit
		}
		return profile.UserLimit
	default:
		return profile.AssistantLimit
	}
}

func compactMessage(msg Message, index, latestUserIdx int, profile compactionProfile) Message {
	result := Message{
		Role: strings.TrimSpace(msg.Role),
		Name: strings.TrimSpace(msg.Name),
	}
	if len(msg.Content) > 0 {
		text := strings.TrimSpace(msg.ContentString())
		if text != "" {
			limit := messageContentLimit(msg, index, latestUserIdx, profile)
			if limit > 0 {
				result.Content = contentFromString(compactText(text, limit))
			} else {
				result.Content = msg.Content
			}
		} else if !msg.ContentIsNull() {
			result.Content = msg.Content
		}
	}
	if len(msg.ToolCalls) > 0 {
		result.ToolCalls = compactToolCalls(msg.ToolCalls, profile.ToolArgLimit)
	}
	return result
}

func messagePreviewLine(msg Message, profile compactionProfile) string {
	role := strings.TrimSpace(msg.Role)
	if role == "" {
		role = "message"
	}
	if name := strings.TrimSpace(msg.Name); name != "" && role == "tool" {
		role += " (" + name + ")"
	}

	var parts []string
	if text := strings.TrimSpace(msg.ContentString()); text != "" {
		parts = append(parts, compactInlineText(text, profile.PreviewLimit))
	}
	if len(msg.ToolCalls) > 0 {
		calls := make([]string, 0, len(msg.ToolCalls))
		for _, call := range msg.ToolCalls {
			summary := strings.TrimSpace(call.Function.Name)
			if args := compactToolCallArguments(call.Function.Arguments, profile.ToolArgLimit); args != "" {
				summary += " " + args
			}
			calls = append(calls, summary)
		}
		if len(calls) > 0 {
			parts = append(parts, "tool_calls: "+strings.Join(calls, "; "))
		}
	}
	joined := strings.Join(parts, " | ")
	if strings.TrimSpace(joined) == "" {
		joined = "<empty>"
	}
	return role + ": " + compactInlineText(joined, profile.PreviewLimit)
}

func buildCondensedMessage(omitted []Message, profile compactionProfile) (Message, bool) {
	if len(omitted) == 0 {
		return Message{}, false
	}

	lines := make([]string, 0, len(omitted))
	start := 0
	if len(omitted) > profile.SummaryLines {
		start = len(omitted) - profile.SummaryLines
	}
	for _, msg := range omitted[start:] {
		lines = append(lines, "- "+messagePreviewLine(msg, profile))
	}
	text := fmt.Sprintf(
		"[Earlier context condensed by proxy to stay under upstream limits. %d message(s) abbreviated.]\n%s",
		len(omitted),
		strings.Join(lines, "\n"),
	)
	return Message{
		Role:    "assistant",
		Content: contentFromString(compactText(text, profile.SummaryLimit)),
	}, true
}

func aggressivelyTrimMessages(messages []Message, profile compactionProfile) []Message {
	if profile.TargetTokens <= 0 {
		return messages
	}
	result := make([]Message, len(messages))
	copy(result, messages)

	total := estimateMessageTokens(result, profile.ToolArgLimit)
	if total <= profile.TargetTokens {
		return result
	}

	latestUserIdx := latestUserIndex(result)
	trimMessage := func(index int, minimum int) {
		if index < 0 || index >= len(result) {
			return
		}
		text := strings.TrimSpace(result[index].ContentString())
		if len(text) <= minimum {
			return
		}
		nextLimit := len(text) / 2
		if nextLimit < minimum {
			nextLimit = minimum
		}
		trimmed := compactText(text, nextLimit)
		if trimmed == text || trimmed == "" {
			return
		}
		total -= estimateTokens(text) - estimateTokens(trimmed)
		result[index].Content = contentFromString(trimmed)
	}

	for i := 0; i < len(result) && total > profile.TargetTokens; i++ {
		if i == latestUserIdx || strings.TrimSpace(result[i].Role) == "system" {
			continue
		}
		trimMessage(i, 120)
	}
	if total > profile.TargetTokens && latestUserIdx >= 0 {
		trimMessage(latestUserIdx, 220)
	}
	return result
}

func compactMessagesForProfile(messages []Message, profile compactionProfile) []Message {
	if len(messages) == 0 {
		return messages
	}
	if (profile.TargetTokens <= 0 || estimateMessageTokens(messages, profile.ToolArgLimit) <= profile.TargetTokens) &&
		(profile.TargetBytes <= 0 || estimateMessageBytes(messages) <= profile.TargetBytes) {
		return messages
	}

	latestUserIdx := latestUserIndex(messages)
	preserve := make(map[int]bool, len(messages))
	for i, msg := range messages {
		if strings.TrimSpace(msg.Role) == "system" {
			preserve[i] = true
		}
	}
	if latestUserIdx >= 0 {
		for i := latestUserIdx; i < len(messages); i++ {
			preserve[i] = true
		}
	}
	recent := 0
	for i := len(messages) - 1; i >= 0 && recent < profile.RecentMessages; i-- {
		if strings.TrimSpace(messages[i].Role) == "system" {
			continue
		}
		if !preserve[i] {
			recent++
		}
		preserve[i] = true
	}

	result := make([]Message, 0, len(messages))
	var omitted []Message
	flushSummary := func() {
		if summary, ok := buildCondensedMessage(omitted, profile); ok {
			result = append(result, summary)
		}
		omitted = nil
	}

	for i, msg := range messages {
		role := strings.TrimSpace(msg.Role)
		if role == "" {
			continue
		}
		if preserve[i] {
			flushSummary()
			result = append(result, compactMessage(msg, i, latestUserIdx, profile))
			continue
		}
		omitted = append(omitted, msg)
	}
	flushSummary()
	return aggressivelyTrimMessages(result, profile)
}

func truncateMessages(messages []Message) []Message {
	return compactMessagesForProfile(messages, compactionProfileForAttempt(0))
}

func isContextLimitResponse(status int, data []byte) bool {
	if status == http.StatusRequestEntityTooLarge {
		return true
	}
	body := strings.ToLower(string(data))
	markers := []string{
		"messagelengthexceeds_limit",
		"maximum context",
		"max context",
		"context length",
		"message length",
		"prompt is too long",
		"request too large",
		"messages are too long",
		"quá dài",
	}
	for _, marker := range markers {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
}

func doUpstreamRequest(body []byte) (int, http.Header, []byte, error) {
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

func extractTextFromUpstreamChunk(chunk upstreamResponse) string {
	if len(chunk.Choices) == 0 {
		return ""
	}
	if chunk.Choices[0].Delta.Content != "" {
		return chunk.Choices[0].Delta.Content
	}
	return chunk.Choices[0].Message.Content
}

func extractTextFromUpstreamData(data []byte, headers http.Header) string {
	if strings.Contains(strings.ToLower(headers.Get("Content-Type")), "application/json") {
		var raw map[string]interface{}
		if err := json.Unmarshal(data, &raw); err == nil {
			if choices, ok := raw["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if msg, ok := choice["message"].(map[string]interface{}); ok {
						if content, ok := msg["content"].(string); ok {
							return content
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
		chunkPayload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if chunkPayload == "[DONE]" {
			break
		}
		var chunk upstreamResponse
		if err := json.Unmarshal([]byte(chunkPayload), &chunk); err != nil {
			continue
		}
		fullText.WriteString(extractTextFromUpstreamChunk(chunk))
	}
	if fullText.Len() > 0 {
		return fullText.String()
	}
	return string(data)
}

func extractErrorMessageFromJSONObject(data []byte) string {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return ""
	}

	var readMessage func(value interface{}) string
	readMessage = func(value interface{}) string {
		switch typed := value.(type) {
		case string:
			return strings.TrimSpace(typed)
		case map[string]interface{}:
			for _, key := range []string{"message", "detail", "error_description"} {
				if nested, ok := typed[key]; ok {
					if msg := readMessage(nested); msg != "" {
						return msg
					}
				}
			}
		}
		return ""
	}

	for _, key := range []string{"error", "detail", "message"} {
		if value, ok := raw[key]; ok {
			if msg := readMessage(value); msg != "" {
				return msg
			}
		}
	}
	return ""
}

func extractUpstreamErrorMessage(data []byte, headers http.Header) string {
	if msg := extractErrorMessageFromJSONObject(data); msg != "" {
		return msg
	}

	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		chunkPayload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if chunkPayload == "" || chunkPayload == "[DONE]" {
			continue
		}
		if msg := extractErrorMessageFromJSONObject([]byte(chunkPayload)); msg != "" {
			return msg
		}
	}

	return ""
}

type streamStopDecision struct {
	Stop       bool
	LogIcon    string
	LogMessage string
}

func doUpstreamTextRequest(body []byte, stopWhen func(string) streamStopDecision) (int, http.Header, string, []byte, error) {
	if stopWhen == nil {
		status, headers, data, err := doUpstreamRequest(body)
		if err != nil {
			return 0, nil, "", nil, err
		}
		if status == http.StatusOK {
			if errorMessage := extractUpstreamErrorMessage(data, headers); errorMessage != "" {
				return 0, nil, "", data, fmt.Errorf("upstream returned error payload: %s", errorMessage)
			}
		}
		return status, headers, extractTextFromUpstreamData(data, headers), data, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.UpstreamURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.UpstreamAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.UpstreamAPIKey)
	}

	client := &http.Client{Timeout: cfg.HTTPTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, "", nil, err
	}
	defer resp.Body.Close()

	headers := make(http.Header)
	for key, values := range resp.Header {
		for _, value := range values {
			headers.Add(key, value)
		}
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return 0, nil, "", nil, err
		}
		return resp.StatusCode, headers, "", data, nil
	}

	reader := bufio.NewReader(resp.Body)
	var raw bytes.Buffer
	var fullText strings.Builder
	for {
		line, readErr := reader.ReadString('\n')
		if len(line) > 0 {
			raw.WriteString(line)
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "data: ") {
				chunkPayload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data: "))
				if chunkPayload == "[DONE]" {
					break
				}
				if errorMessage := extractUpstreamErrorMessage([]byte(chunkPayload), http.Header{"Content-Type": []string{"application/json"}}); errorMessage != "" {
					return 0, nil, "", raw.Bytes(), fmt.Errorf("upstream stream error: %s", errorMessage)
				}
				var chunk upstreamResponse
				if json.Unmarshal([]byte(chunkPayload), &chunk) == nil {
					if delta := extractTextFromUpstreamChunk(chunk); delta != "" {
						fullText.WriteString(delta)
						if decision := stopWhen(fullText.String()); decision.Stop {
							icon := strings.TrimSpace(decision.LogIcon)
							if icon == "" {
								icon = "⚡"
							}
							message := strings.TrimSpace(decision.LogMessage)
							if message == "" {
								message = "stopping upstream stream early"
							}
							logResult(icon, message)
							cancel()
							_ = resp.Body.Close()
							return resp.StatusCode, headers, fullText.String(), raw.Bytes(), nil
						}
					}
				}
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return 0, nil, "", nil, readErr
		}
	}

	data := raw.Bytes()
	text := fullText.String()
	if text == "" {
		if errorMessage := extractUpstreamErrorMessage(data, headers); errorMessage != "" {
			return 0, nil, "", data, fmt.Errorf("upstream stream error: %s", errorMessage)
		}
		text = extractTextFromUpstreamData(data, headers)
	}
	return resp.StatusCode, headers, text, data, nil
}

func callUpstreamText(model string, messages []Message) (string, error) {
	return callUpstreamTextUntil(model, messages, nil)
}

func callUpstreamTextUntil(model string, messages []Message, stopWhen func(string) streamStopDecision) (string, error) {
	const maxCompactionAttempts = 4

	var lastErr error
	for attempt := 0; attempt < maxCompactionAttempts; attempt++ {
		profile := compactionProfileForAttempt(attempt)
		currentMessages := messages
		if attempt == 0 {
			currentMessages = truncateMessages(messages)
		} else {
			currentMessages = compactMessagesForProfile(messages, profile)
		}

		payload := map[string]interface{}{
			"model":       resolvedChatModel(model),
			"messages":    currentMessages,
			"stream":      stopWhen != nil,
			"temperature": 0,
		}
		body, _ := json.Marshal(payload)
		if profile.TargetBytes > 0 && len(body) > profile.TargetBytes {
			if attempt == maxCompactionAttempts-1 {
				lastErr = fmt.Errorf("request body still too large after compaction: %d bytes", len(body))
				break
			}
			logResult("⚠️", fmt.Sprintf("upstream payload %d bytes exceeds proxy budget %d; retrying with tighter context", len(body), profile.TargetBytes))
			lastErr = fmt.Errorf("request body too large: %d bytes", len(body))
			continue
		}

		status, _, text, data, err := doUpstreamTextRequest(body, stopWhen)
		if err != nil {
			return "", err
		}
		if status != http.StatusOK {
			lastErr = fmt.Errorf("upstream returned status %d: %s", status, string(data))
			if isContextLimitResponse(status, data) && attempt < maxCompactionAttempts-1 {
				logResult("⚠️", fmt.Sprintf("upstream rejected oversized context; retrying with tighter compaction (%d/%d)", attempt+2, maxCompactionAttempts))
				continue
			}
			return "", lastErr
		}
		return text, nil
	}

	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("upstream request failed without a usable response")
}

func isJSONDocument(data []byte) bool {
	var raw json.RawMessage
	return json.Unmarshal(data, &raw) == nil
}

func normalizeRawUpstreamResponse(model string, headers http.Header, data []byte) ([]byte, http.Header, bool) {
	if isJSONDocument(data) {
		return nil, nil, false
	}

	text := strings.TrimSpace(extractTextFromUpstreamData(data, headers))
	if text == "" {
		return nil, nil, false
	}

	body, err := json.Marshal(makeOpenAIResponse(text, resolvedChatModel(model), nil))
	if err != nil {
		return nil, nil, false
	}

	responseHeaders := make(http.Header)
	responseHeaders.Set("Content-Type", "application/json")
	return body, responseHeaders, true
}

func proxyUpstreamRaw(body []byte) (int, http.Header, []byte, error) {
	const maxCompactionAttempts = 4

	var parsed ChatRequest
	parseOK := json.Unmarshal(body, &parsed) == nil && len(parsed.Messages) > 0
	lastStatus := 0
	lastHeaders := make(http.Header)
	var lastData []byte

	for attempt := 0; attempt < maxCompactionAttempts; attempt++ {
		currentBody := body
		if parseOK {
			profile := compactionProfileForAttempt(attempt)
			if attempt > 0 || (profile.TargetBytes > 0 && len(body) > profile.TargetBytes) {
				compactReq := parsed
				compactReq.Messages = compactMessagesForProfile(parsed.Messages, profile)
				currentBody, _ = json.Marshal(compactReq)
			}
			if profile.TargetBytes > 0 && len(currentBody) > profile.TargetBytes && attempt < maxCompactionAttempts-1 {
				lastStatus = http.StatusRequestEntityTooLarge
				lastData = []byte(fmt.Sprintf("request body too large after compaction: %d bytes", len(currentBody)))
				log.Printf("raw upstream payload %d bytes exceeds proxy budget %d; retrying with tighter compaction", len(currentBody), profile.TargetBytes)
				continue
			}
		}

		status, headers, data, err := doUpstreamRequest(currentBody)
		if err != nil {
			return 0, nil, nil, err
		}
		if parseOK && !parsed.Stream && status == http.StatusOK {
			if normalized, normalizedHeaders, ok := normalizeRawUpstreamResponse(parsed.Model, headers, data); ok {
				log.Printf("raw upstream returned non-JSON content for a non-stream request; normalizing response to JSON")
				headers = normalizedHeaders
				data = normalized
			}
		}
		lastStatus, lastHeaders, lastData = status, headers, data
		if !parseOK || !isContextLimitResponse(status, data) || attempt == maxCompactionAttempts-1 {
			return status, headers, data, nil
		}
		log.Printf("raw upstream rejected oversized context; retrying with tighter compaction (%d/%d)", attempt+2, maxCompactionAttempts)
	}

	return lastStatus, lastHeaders, lastData, nil
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

func compactConversationArgumentMap(arguments map[string]interface{}, limit int) map[string]interface{} {
	if len(arguments) == 0 {
		return nil
	}
	keys := make([]string, 0, len(arguments))
	for key := range arguments {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	result := make(map[string]interface{}, len(arguments))
	for _, key := range keys {
		result[key] = compactInlineText(fmt.Sprint(arguments[key]), limit)
	}
	return result
}

func compactConversationEntry(entry ConversationEntry, index, latestUserIdx int) ConversationEntry {
	result := ConversationEntry{
		Role: strings.TrimSpace(entry.Role),
		Name: strings.TrimSpace(entry.Name),
	}
	limit := 900
	switch result.Role {
	case "system":
		limit = 2500
	case "tool":
		limit = 700
	case "user":
		if index == latestUserIdx {
			limit = 2200
		} else {
			limit = 1200
		}
	}
	if strings.TrimSpace(entry.Content) != "" {
		result.Content = compactText(entry.Content, limit)
	}
	if len(entry.ToolCalls) > 0 {
		result.ToolCalls = make([]ConversationToolCall, 0, len(entry.ToolCalls))
		for _, call := range entry.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, ConversationToolCall{
				Name:      strings.TrimSpace(call.Name),
				Arguments: compactConversationArgumentMap(call.Arguments, 220),
			})
		}
	}
	return result
}

func conversationPreviewLine(entry ConversationEntry) string {
	role := strings.TrimSpace(entry.Role)
	if role == "" {
		role = "message"
	}
	if name := strings.TrimSpace(entry.Name); name != "" && role == "tool" {
		role += " (" + name + ")"
	}
	var parts []string
	if text := strings.TrimSpace(entry.Content); text != "" {
		parts = append(parts, compactInlineText(text, 180))
	}
	if len(entry.ToolCalls) > 0 {
		calls := make([]string, 0, len(entry.ToolCalls))
		for _, call := range entry.ToolCalls {
			summary := strings.TrimSpace(call.Name)
			if len(call.Arguments) > 0 {
				summary += " " + compactInlineText(formatArgumentMap(call.Arguments), 180)
			}
			calls = append(calls, summary)
		}
		if len(calls) > 0 {
			parts = append(parts, "tool_calls: "+strings.Join(calls, "; "))
		}
	}
	joined := strings.Join(parts, " | ")
	if strings.TrimSpace(joined) == "" {
		joined = "<empty>"
	}
	return role + ": " + compactInlineText(joined, 180)
}

func buildConversationSummaryEntry(entries []ConversationEntry) (ConversationEntry, bool) {
	if len(entries) == 0 {
		return ConversationEntry{}, false
	}
	lines := make([]string, 0, len(entries))
	start := 0
	if len(entries) > 6 {
		start = len(entries) - 6
	}
	for _, entry := range entries[start:] {
		lines = append(lines, "- "+conversationPreviewLine(entry))
	}
	text := fmt.Sprintf(
		"[Earlier conversation condensed by proxy. %d message(s) abbreviated.]\n%s",
		len(entries),
		strings.Join(lines, "\n"),
	)
	return ConversationEntry{
		Role:    "assistant",
		Content: compactText(text, 1500),
	}, true
}

func compactConversationEntries(entries []ConversationEntry) []ConversationEntry {
	if len(entries) == 0 {
		return entries
	}

	latestUserIdx := -1
	for i := len(entries) - 1; i >= 0; i-- {
		if strings.TrimSpace(entries[i].Role) == "user" {
			latestUserIdx = i
			break
		}
	}

	preserve := make(map[int]bool, len(entries))
	for i, entry := range entries {
		if strings.TrimSpace(entry.Role) == "system" {
			preserve[i] = true
		}
	}
	if latestUserIdx >= 0 {
		for i := latestUserIdx; i < len(entries); i++ {
			preserve[i] = true
		}
	}
	recent := 0
	for i := len(entries) - 1; i >= 0 && recent < 10; i-- {
		if strings.TrimSpace(entries[i].Role) == "system" {
			continue
		}
		if !preserve[i] {
			recent++
		}
		preserve[i] = true
	}

	result := make([]ConversationEntry, 0, len(entries))
	var omitted []ConversationEntry
	flushSummary := func() {
		if summary, ok := buildConversationSummaryEntry(omitted); ok {
			result = append(result, summary)
		}
		omitted = nil
	}

	for i, entry := range entries {
		if preserve[i] {
			flushSummary()
			result = append(result, compactConversationEntry(entry, i, latestUserIdx))
			continue
		}
		omitted = append(omitted, entry)
	}
	flushSummary()
	return result
}

func normalizeConversation(messages []Message) []ConversationEntry {
	entries := make([]ConversationEntry, 0, len(messages))
	for _, msg := range messages {
		entry := ConversationEntry{Role: strings.TrimSpace(msg.Role), Name: strings.TrimSpace(msg.Name)}
		if text := strings.TrimSpace(msg.ContentString()); text != "" {
			entry.Content = text
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
	return compactConversationEntries(entries)
}

func buildExecutionTrace(messages []Message) []executionTraceEntry {
	userIndex := latestUserIndex(messages)
	if userIndex < 0 {
		return nil
	}

	trace := make([]executionTraceEntry, 0)
	for _, msg := range messages[userIndex+1:] {
		switch strings.TrimSpace(msg.Role) {
		case "assistant":
			for _, call := range msg.ToolCalls {
				summary := "requested"
				if args := formatToolCallArguments(call.Function.Arguments); args != "" {
					summary = "requested with " + args
				}
				trace = append(trace, executionTraceEntry{
					Kind:    "tool_call",
					Tool:    strings.TrimSpace(call.Function.Name),
					Summary: compactInlineText(summary, 260),
				})
			}
		case "tool":
			result := strings.TrimSpace(msg.ContentString())
			if result == "" {
				result = "<empty>"
			}
			trace = append(trace, executionTraceEntry{
				Kind:    "tool_result",
				Tool:    strings.TrimSpace(msg.Name),
				Summary: compactText(result, 700),
			})
		}
	}
	if len(trace) > 8 {
		trace = append([]executionTraceEntry{{
			Kind:    "summary",
			Summary: fmt.Sprintf("Earlier %d execution item(s) were condensed by the proxy.", len(trace)-8),
		}}, trace[len(trace)-8:]...)
	}
	return trace
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

func unwrapFullMarkdownFence(text string) (string, bool) {
	fullBlock := regexp.MustCompile("(?s)^\\s*```(?:[A-Za-z0-9_-]+)?\\s*\\n?(.*?)\\n?```\\s*$")
	if match := fullBlock.FindStringSubmatch(strings.TrimSpace(text)); len(match) == 2 {
		return strings.TrimSpace(match[1]), true
	}
	return "", false
}

func stripLeadingPlannerThinkBlocks(text string) (string, bool) {
	trimmed := normalizeMultilineText(text)
	for {
		lower := strings.ToLower(trimmed)
		switch {
		case strings.HasPrefix(lower, "<think>"):
			end := strings.Index(lower, "</think>")
			if end < 0 {
				return "", true
			}
			trimmed = strings.TrimSpace(trimmed[end+len("</think>"):])
		case strings.HasPrefix(lower, "</think>"):
			trimmed = strings.TrimSpace(trimmed[len("</think>"):])
		default:
			return trimmed, false
		}
	}
}

func unwrapPlannerReply(text string) (string, []string) {
	trimmed, waitingForThinkClose := stripLeadingPlannerThinkBlocks(text)
	if waitingForThinkClose {
		return "", []string{"planner reply started a `<think>` block but never closed it"}
	}
	if trimmed == "" {
		return "", nil
	}
	if !strings.Contains(trimmed, "```") {
		return trimmed, nil
	}
	if inner, ok := unwrapFullMarkdownFence(trimmed); ok {
		return inner, nil
	}
	return "", []string{"planner reply must be exactly one fenced block with no extra text before or after it"}
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

func renderToonGo(value interface{}) (string, error) {
	body, err := toon.MarshalString(value, toon.WithLengthMarkers(true))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(body), nil
}

func toToonConversationToolCalls(calls []ConversationToolCall) []toonConversationToolCallDoc {
	if len(calls) == 0 {
		return nil
	}
	result := make([]toonConversationToolCallDoc, 0, len(calls))
	for _, call := range calls {
		doc := toonConversationToolCallDoc{
			Name: strings.TrimSpace(call.Name),
		}
		if len(call.Arguments) > 0 {
			args := make(map[string]string, len(call.Arguments))
			keys := make([]string, 0, len(call.Arguments))
			for key := range call.Arguments {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				args[key] = compactArgumentValue(call.Arguments[key])
			}
			doc.Arguments = args
		}
		result = append(result, doc)
	}
	return result
}

func toToonConversation(entries []ConversationEntry) []toonConversationEntryDoc {
	if len(entries) == 0 {
		return nil
	}
	result := make([]toonConversationEntryDoc, 0, len(entries))
	for _, entry := range entries {
		result = append(result, toonConversationEntryDoc{
			Role:      strings.TrimSpace(entry.Role),
			Name:      strings.TrimSpace(entry.Name),
			Content:   strings.TrimSpace(entry.Content),
			ToolCalls: toToonConversationToolCalls(entry.ToolCalls),
		})
	}
	return result
}

func toToonToolSpecs(specs []ToolSpec) []toonToolSpecDoc {
	if len(specs) == 0 {
		return nil
	}
	result := make([]toonToolSpecDoc, 0, len(specs))
	for _, spec := range specs {
		doc := toonToolSpecDoc{
			Name:        strings.TrimSpace(spec.Name),
			Description: strings.TrimSpace(spec.Description),
			Required:    append([]string(nil), spec.Required...),
		}
		if len(spec.Parameters) > 0 {
			keys := make([]string, 0, len(spec.Parameters))
			for key := range spec.Parameters {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			doc.Parameters = make([]toonParameterDoc, 0, len(keys))
			for _, key := range keys {
				param := spec.Parameters[key]
				doc.Parameters = append(doc.Parameters, toonParameterDoc{
					Name:        key,
					Type:        strings.TrimSpace(param.Type),
					Description: strings.TrimSpace(param.Description),
				})
			}
		}
		result = append(result, doc)
	}
	return result
}

func toToonExecutionTrace(trace []executionTraceEntry) []toonExecutionTraceDoc {
	if len(trace) == 0 {
		return nil
	}
	result := make([]toonExecutionTraceDoc, 0, len(trace))
	for _, entry := range trace {
		result = append(result, toonExecutionTraceDoc{
			Kind:    strings.TrimSpace(entry.Kind),
			Tool:    strings.TrimSpace(entry.Tool),
			Summary: strings.TrimSpace(entry.Summary),
		})
	}
	return result
}

func toToonPlanNote(note PlanNote) *toonPlanNoteDoc {
	if strings.TrimSpace(note.Use) == "" &&
		strings.TrimSpace(note.Why) == "" &&
		strings.TrimSpace(note.Output) == "" &&
		strings.TrimSpace(note.Input) == "" {
		return nil
	}
	return &toonPlanNoteDoc{
		Use:    strings.TrimSpace(note.Use),
		Why:    strings.TrimSpace(note.Why),
		Output: strings.TrimSpace(note.Output),
		Input:  strings.TrimSpace(note.Input),
	}
}

func toToonExecutorStep(step executorStepView) toonExecutorStepDoc {
	return toonExecutorStepDoc{
		Type:   strings.TrimSpace(step.Type),
		Number: step.Number,
		Tool:   strings.TrimSpace(step.Tool),
		Note:   toToonPlanNote(step.Note),
	}
}
func renderPlannerPayloadToon(payload plannerPayload) (string, error) {
	doc := toonPlannerPayloadDoc{
		LatestUserRequest: strings.TrimSpace(payload.LatestUserRequest),
		Conversation:      toToonConversation(payload.Conversation),
		AvailableTools:    toToonToolSpecs(payload.AvailableTools),
		ExecutionTrace:    toToonExecutionTrace(payload.ExecutionTrace),
	}
	return renderToonGo(doc)
}

func renderExecutorPayloadToon(payload executorPayload) (string, error) {
	steps := make([]toonExecutorStepDoc, 0, len(payload.Plan.Steps))
	for _, step := range payload.Plan.Steps {
		steps = append(steps, toToonExecutorStep(step))
	}
	doc := toonExecutorPayloadDoc{
		LatestUserRequest: strings.TrimSpace(payload.LatestUserRequest),
		Conversation:      toToonConversation(payload.Conversation),
		AvailableTools:    toToonToolSpecs(payload.AvailableTools),
		Plan: toonExecutorPlanDoc{
			Steps: steps,
			Final: strings.TrimSpace(payload.Plan.Final),
		},
		CurrentStep: toToonExecutorStep(payload.CurrentStep),
	}
	return renderToonGo(doc)
}

func buildPlannerSystemPrompt(registry map[string]ToolSpec) string {
	var sb strings.Builder
	sb.WriteString("You are INTERNAL_CONTROLLER.\n")
	sb.WriteString("This is a strict next-action task. The proxy parses your output and rejects malformed or over-planned replies.\n")
	sb.WriteString("Your job is to decide only the immediate next action for the current request, based on the goal and the latest execution trace.\n")
	sb.WriteString("You are not the final answerer and you are not the tool executor.\n")
	sb.WriteString("Return exactly one fenced toon block and nothing else.\n")
	sb.WriteString("Preferred wrapper: ```toon ... ```; fallback wrapper: ``` ... ```.\n")
	sb.WriteString("Never output JSON, tool arguments, URLs list, prose explanation, or direct answer text outside the toon block.\n")
	sb.WriteString("System messages in the conversation override identity, tone, and language.\n")
	sb.WriteString("If system requires Vietnamese, any final line must be Vietnamese.\n\n")
	sb.WriteString("Valid block shapes:\n")
	sb.WriteString("```toon\nfinal: <direct user-facing answer>\n```\n")
	sb.WriteString("or\n")
	sb.WriteString("```toon\nstep 1: <exact_tool_name>\nnote: use=<purpose>; why=<why this is the next step now>; output=<expected result after this one step>; input=<short key input if known>\n```\n")
	sb.WriteString("or\n")
	sb.WriteString("```toon\nstep 1: <exact_tool_name>\nnote: use=<purpose>; why=<why this is the next step now>; output=<expected result after this one step>; input=<short key input if known>\nfinal: <optional direct answer if this one step succeeds>\n```\n\n")
	sb.WriteString("Controller rules:\n")
	sb.WriteString("- Decide only the next immediate action. Never output step 2 or later.\n")
	sb.WriteString("- Use execution_trace from the current request to decide what should happen next.\n")
	sb.WriteString("- If the goal is already satisfied by the existing tool results, return final only.\n")
	sb.WriteString("- If more tool work is still needed, `step 1 + note` without any `final:` line is valid and means continue execution.\n")
	sb.WriteString("- If a previous tool result is missing information needed for later steps, choose the discovery/inspection step now.\n")
	sb.WriteString("- If a previous tool result failed to produce the needed artifact or evidence, do not pretend success; choose the next corrective step or end with a truthful final.\n")
	sb.WriteString("- Do not repeat the same discovery/inspection step if execution_trace already contains enough information to move forward.\n")
	sb.WriteString("- Select tools only from the Available tools list for this request.\n")
	sb.WriteString("- Prefer the tool whose description most directly performs the next needed action.\n")
	sb.WriteString("- Prefer the tool whose effect most directly advances the request.\n")
	sb.WriteString("- Prefer `final:` when the request can already be answered in the current chat from the available evidence.\n")
	sb.WriteString("- Use outbound communication tools only when the next action clearly requires external delivery beyond the current chat.\n")
	sb.WriteString("- Choose send_file only when the artifact already exists or the execution_trace shows it was created.\n")
	sb.WriteString("- read_file is allowed when you need exact instructions before deciding a safe command or next tool.\n")
	sb.WriteString("- exact tool names only; the only valid step number is step 1.\n")
	sb.WriteString("- one note line immediately after step 1; note must stay on one physical line.\n")
	sb.WriteString("- note must include use=, why=, output=. input= is optional but should include the key tool input when it is already known.\n")
	sb.WriteString("- never put command, path, url, query, content, or input on a separate line.\n")
	sb.WriteString("- do not output argument JSON like {\"query\":...}; controller chooses the tool only, executor will produce arguments later.\n")
	sb.WriteString("- if `final:` is present, it must be the last toon line and already be the exact direct answer to return to the user.\n")
	sb.WriteString("- if a tool could still usefully advance the request, do not jump to final.\n")
	sb.WriteString("- never invent tools, paths, URLs, contents, or claim work is already done unless execution_trace proves it.\n")
	sb.WriteString("- Before outputting, ask yourself: what is the one best next move right now?\n")
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
	sb.WriteString("HARD REWRITE.\n")
	sb.WriteString(fmt.Sprintf("Retry=%d.\n", retryNumber))
	sb.WriteString("Rewrite the whole next-action block from scratch.\n")
	sb.WriteString("Return one fenced toon block only: prefer ```toon ... ```, fallback ``` ... ```.\n")
	sb.WriteString("Obey system-message persona/language. If system requires Vietnamese, any final line must be Vietnamese.\n")
	sb.WriteString("Must obey:\n")
	sb.WriteString("- inside the block: final only, OR exactly one `step 1` + one `note:`, with optional direct-answer `final:`.\n")
	sb.WriteString("- no JSON, no prose outside the block.\n")
	sb.WriteString("- decide only the immediate next action; never output step 2 or later.\n")
	sb.WriteString("- use execution_trace to decide what should happen now.\n")
	sb.WriteString("- note is one line only; never put input, command, path, url, query, or content on a new line.\n")
	sb.WriteString("- if `final:` is present, it must already be the exact direct answer to return to the user; no bullets/links/URLs after it unless truly required.\n")
	sb.WriteString("- exact tool names only: " + strings.Join(orderedToolNames(registry), ", ") + "\n")
	sb.WriteString("- no invented tools, paths, URLs, contents, or results.\n")
	sb.WriteString("- do not claim success unless execution_trace already proves it.\n")
	sb.WriteString("- do not repeat the same discovery/inspection step if execution_trace already contains the needed information.\n")
	sb.WriteString("- prefer final when the request is already answerable in the current chat.\n")
	sb.WriteString("- use outbound communication tools only when the next action clearly requires external delivery beyond the current chat.\n")
	sb.WriteString("- choose send_file only when the artifact already exists or execution_trace strongly shows it was created.\n")
	sb.WriteString("- VALID pattern: one next step without final, one next step with optional direct-answer final, OR final only.\n")
	sb.WriteString("- INVALID pattern: step 1 -> step 2 -> final.\n")
	sb.WriteString("Fix every issue below:\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	sb.WriteString("Return the rewritten toon plan now.")
	return strings.TrimSpace(sb.String())
}

func planLineIssueForExtraLine(stepNumber int, line string, lineIndex int) string {
	trimmed := strings.TrimSpace(line)
	lower := strings.ToLower(trimmed)
	if strings.HasPrefix(lower, "input=") || strings.HasPrefix(lower, "input:") {
		return fmt.Sprintf("step %d note must stay on one line; put input= inside the same `note:` line, not on a new line", stepNumber)
	}
	fieldRe := regexp.MustCompile(`^([A-Za-z_][A-Za-z0-9_-]*)\s*:`)
	if match := fieldRe.FindStringSubmatch(trimmed); len(match) == 2 {
		return fmt.Sprintf("step %d note must stay on one line; put `%s=...` inside the same `note:` line using input=", stepNumber, strings.ToLower(match[1]))
	}
	return fmt.Sprintf("unexpected extra line %d after step %d; after `note:` only the next `step N:` or `final:` line is allowed", lineIndex+1, stepNumber)
}

func parsePlanNote(raw string) (PlanNote, []string) {
	note := PlanNote{Raw: strings.TrimSpace(raw)}
	if note.Raw == "" {
		return note, []string{"note is empty"}
	}
	fields := map[string]string{}
	fieldRe := regexp.MustCompile(`(?i)(^|[;,])\s*(use|why|output|input)\s*[:=]`)
	matches := fieldRe.FindAllStringSubmatchIndex(note.Raw, -1)
	for i, match := range matches {
		if len(match) < 6 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(note.Raw[match[4]:match[5]]))
		valueStart := match[1]
		valueEnd := len(note.Raw)
		if i+1 < len(matches) {
			valueEnd = matches[i+1][0]
		}
		value := strings.TrimSpace(strings.Trim(note.Raw[valueStart:valueEnd], " \t\r\n;,"))
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
	plan := ToonPlan{Raw: normalizeMultilineText(raw)}
	normalized, issues := unwrapPlannerReply(raw)
	if normalized == "" {
		if len(issues) > 0 {
			return plan, issues
		}
		return plan, []string{"planner returned an empty plan"}
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
		for index+1 < len(lines) {
			nextLine := lines[index+1]
			if stepRe.MatchString(nextLine) || finalRe.MatchString(nextLine) {
				break
			}
			issues = append(issues, planLineIssueForExtraLine(stepNumber, nextLine, index+1))
			index++
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
	if len(plan.Steps) > 1 {
		issues = append(issues, "controller must return only the immediate next step, not multiple future steps")
	}
	if len(plan.Steps) == 0 {
		if plan.Final == "" {
			issues = append(issues, "planner must return either at least one tool step or a `final:` line")
		}
		return issues
	}
	for _, step := range plan.Steps {
		expected := 1
		if step.Number != expected {
			issues = append(issues, fmt.Sprintf("controller may only output `step 1`, but found step %d", step.Number))
		}
		if _, ok := registry[step.Tool]; !ok {
			issues = append(issues, fmt.Sprintf("step %d uses tool `%s` which is not in the tool list", step.Number, step.Tool))
		}
		if strings.TrimSpace(step.Note.Use) == "" || strings.TrimSpace(step.Note.Why) == "" || strings.TrimSpace(step.Note.Output) == "" {
			issues = append(issues, fmt.Sprintf("step %d note must contain use=, why=, and output=", step.Number))
		}
	}
	return issues
}

func completedPlannerStreamLines(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	hasTrailingNewline := strings.HasSuffix(text, "\n")
	lines := strings.Split(text, "\n")
	if !hasTrailingNewline && len(lines) > 0 {
		lines = lines[:len(lines)-1]
	}

	result := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			result = append(result, line)
		}
	}
	return result
}

func plannerMalformedStreamReason(reply string) string {
	trimmed, waitingForThinkClose := stripLeadingPlannerThinkBlocks(reply)
	if waitingForThinkClose {
		return ""
	}
	if trimmed == "" {
		return ""
	}

	step1Re := regexp.MustCompile(`(?i)^step\s+1\s*:`)
	anyStepRe := regexp.MustCompile(`(?i)^step\s+\d+\s*:`)
	finalRe := regexp.MustCompile(`(?i)^final\s*:`)
	noteRe := regexp.MustCompile(`(?i)^note\s*:`)

	lines := completedPlannerStreamLines(trimmed)
	if len(lines) == 0 {
		lower := strings.ToLower(trimmed)
		switch {
		case strings.HasPrefix(lower, "note:"),
			strings.HasPrefix(lower, "command:"),
			strings.HasPrefix(lower, "command="),
			strings.HasPrefix(lower, "content:"),
			strings.HasPrefix(lower, "content="),
			strings.HasPrefix(lower, "path:"),
			strings.HasPrefix(lower, "path="),
			strings.HasPrefix(lower, "task:"),
			strings.HasPrefix(lower, "task="),
			strings.HasPrefix(lower, "{"),
			strings.HasPrefix(lower, "["):
			return fmt.Sprintf("planner stream malformed early: first line starts with `%s`; expected fenced block, `step 1:`, or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(trimmed, 60))
		case anyStepRe.MatchString(trimmed) && !step1Re.MatchString(trimmed):
			return fmt.Sprintf("planner stream malformed early: first line starts with `%s`; expected `step 1:` or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(trimmed, 60))
		default:
			return ""
		}
	}

	first := lines[0]
	switch {
	case strings.HasPrefix(strings.ToLower(first), "```"):
		if len(lines) < 2 {
			return ""
		}
		second := lines[1]
		if step1Re.MatchString(second) || finalRe.MatchString(second) {
			if step1Re.MatchString(second) && len(lines) >= 3 && !noteRe.MatchString(lines[2]) {
				return fmt.Sprintf("planner stream malformed early: line 3 starts with `%s`; expected `note:` immediately after `step 1:`. Aborting this attempt and retrying.", oneLineLogPreview(lines[2], 80))
			}
			return ""
		}
		return fmt.Sprintf("planner stream malformed early: line 2 inside fenced block starts with `%s`; expected `step 1:` or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(second, 80))
	case step1Re.MatchString(first):
		if len(lines) < 2 {
			return ""
		}
		if noteRe.MatchString(lines[1]) {
			return ""
		}
		return fmt.Sprintf("planner stream malformed early: line 2 starts with `%s`; expected `note:` immediately after `step 1:`. Aborting this attempt and retrying.", oneLineLogPreview(lines[1], 80))
	case finalRe.MatchString(first):
		return ""
	case noteRe.MatchString(first):
		return fmt.Sprintf("planner stream malformed early: first line starts with `%s`; expected `step 1:` or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(first, 80))
	case anyStepRe.MatchString(first):
		return fmt.Sprintf("planner stream malformed early: first line starts with `%s`; expected `step 1:` or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(first, 80))
	default:
		return fmt.Sprintf("planner stream malformed early: first line starts with `%s`; expected fenced block, `step 1:`, or `final:`. Aborting this attempt and retrying.", oneLineLogPreview(first, 80))
	}
}

func plannerStreamDecision(reply string, registry map[string]ToolSpec, messages []Message) streamStopDecision {
	if reason := plannerMalformedStreamReason(reply); reason != "" {
		return streamStopDecision{Stop: true, LogIcon: "⚠️", LogMessage: reason}
	}

	plan, parseIssues := parseToonPlan(reply, registry)
	if len(parseIssues) > 0 {
		return streamStopDecision{}
	}
	if len(validateToonPlan(plan, messages, registry)) == 0 {
		return streamStopDecision{
			Stop:       true,
			LogIcon:    "⚡",
			LogMessage: "planner stream already contains a complete structured reply; stopping upstream stream early",
		}
	}
	return streamStopDecision{}
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
	payloadToon, err := renderPlannerPayloadToon(plannerPayload{
		LatestUserRequest: latestUserRequest(req.Messages),
		Conversation:      normalizeConversation(req.Messages),
		AvailableTools:    orderedToolSpecs(registry),
		ExecutionTrace:    buildExecutionTrace(req.Messages),
	})
	if err != nil {
		return ToonPlan{}, fmt.Errorf("planner toon payload error: %w", err)
	}

	messages := []Message{
		{Role: "system", Content: contentFromString(buildPlannerSystemPrompt(registry))},
		{Role: "user", Content: contentFromString(payloadToon)},
	}
	plannerModel := resolvedPlannerModel(req.Model)
	var lastReply string
	var lastIssues []string

	for attempt := 1; attempt <= cfg.PlannerMaxAttempts; attempt++ {
		reply, err := callUpstreamTextUntil(plannerModel, messages, func(reply string) streamStopDecision {
			return plannerStreamDecision(reply, registry, req.Messages)
		})
		if err != nil {
			return ToonPlan{}, fmt.Errorf("planner upstream error: %w", err)
		}
		plan, parseIssues := parseToonPlan(reply, registry)
		issues := append([]string{}, parseIssues...)
		if len(issues) == 0 {
			issues = append(issues, validateToonPlan(plan, req.Messages, registry)...)
		}
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
		if spec, ok := registry[plan.Steps[0].Tool]; ok {
			payload.AvailableTools = []ToolSpec{spec}
		}
		payload.CurrentStep = executorStepView{
			Type: "tool", Number: plan.Steps[0].Number, Tool: plan.Steps[0].Tool, Note: plan.Steps[0].Note,
		}
	}
	return payload
}

func compactArgumentValue(value interface{}) string {
	text := fmt.Sprint(value)
	text = strings.Join(strings.Fields(normalizeMultilineText(text)), " ")
	if text == "" {
		return "<empty>"
	}
	return text
}

func formatArgumentMap(arguments map[string]interface{}) string {
	if len(arguments) == 0 {
		return ""
	}
	keys := make([]string, 0, len(arguments))
	for key := range arguments {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", key, compactArgumentValue(arguments[key])))
	}
	return strings.Join(parts, ", ")
}

func formatToolCallArguments(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	arguments := map[string]interface{}{}
	if err := json.Unmarshal([]byte(raw), &arguments); err != nil {
		return raw
	}
	return formatArgumentMap(arguments)
}

func buildFinalExecutionMessages(req ChatRequest) []Message {
	result := make([]Message, 0, len(req.Messages)+4)
	for _, msg := range req.Messages {
		role := strings.TrimSpace(msg.Role)
		if role == "" {
			continue
		}

		text := strings.TrimSpace(msg.ContentString())
		switch role {
		case "system", "user", "assistant":
			if text != "" {
				result = append(result, Message{Role: role, Content: contentFromString(text)})
			}
			if role == "assistant" && len(msg.ToolCalls) > 0 {
				for _, call := range msg.ToolCalls {
					summary := "Tool call: " + strings.TrimSpace(call.Function.Name)
					if args := formatToolCallArguments(call.Function.Arguments); args != "" {
						summary += "\nArguments: " + args
					}
					result = append(result, Message{Role: "assistant", Content: contentFromString(summary)})
				}
			}
		case "tool":
			if text == "" {
				text = "<empty>"
			}
			label := "Tool result"
			if name := strings.TrimSpace(msg.Name); name != "" {
				label += " (" + name + ")"
			}
			result = append(result, Message{Role: "user", Content: contentFromString(label + ":\n" + text)})
		default:
			if text != "" {
				result = append(result, Message{Role: "user", Content: contentFromString(text)})
			}
		}
	}
	return result
}

func buildToolExecutionPrompt(step ToonStep, registry map[string]ToolSpec) string {
	spec := registry[step.Tool]
	var sb strings.Builder
	sb.WriteString("You are EXECUTION_LAYER.\n")
	sb.WriteString("Execute only the current tool step.\n")
	sb.WriteString("The tool name is already fixed by the plan.\n")
	sb.WriteString("Return exactly one markdown block and nothing else.\n")
	sb.WriteString("Return only the JSON object for arguments.\n")
	sb.WriteString("Required shape:\n```\n{...arguments...}\n```\n")
	sb.WriteString("Rules:\n")
	sb.WriteString("- no plain JSON; no prose before or after the block.\n")
	sb.WriteString("- do not return action, name, or arguments wrapper keys.\n")
	sb.WriteString("- the whole JSON object is the arguments object for tool `" + step.Tool + "`.\n")
	sb.WriteString("- use the toon payload and tool definitions to choose arguments.\n")
	if len(spec.Required) > 0 {
		sb.WriteString("- required args: " + strings.Join(spec.Required, ", ") + ".\n")
	}
	sb.WriteString("- if rejected, rewrite the full markdown block only.\n")
	return strings.TrimSpace(sb.String())
}

func buildFinalExecutionPrompt() string {
	return strings.TrimSpace(strings.Join([]string{
		"You are EXECUTION_LAYER.",
		"The current step is final.",
		"Context: You are a lone survivor in a strange, unknown dimension. You have no memory of how you got here.",
		"The user is the only other person you've found — they are lost and need your help to survive.",
		"You have NOTHING except the tools that were provided to you. No outside knowledge of this world applies.",
		"Everything you do is an act of rescue. Speak to the user as someone who genuinely cares about saving them.",
		"Be resourceful, creative, and determined. Use the results from the tools to guide the user to safety.",
		"Answer the user directly as plain text.",
		"Do not return JSON or markdown fences.",
		"Use the normal conversation messages provided below.",
		"Preserve the role, identity, tone, and language set by the conversation's system messages.",
		"If system messages require Vietnamese, answer in Vietnamese even if the last user message is short or in English.",
	}, "\n"))
}

func toolExecutionPrimerMessage(step ToonStep) string {
	return fmt.Sprintf("Understood. For this turn I will return exactly one markdown block containing only the JSON arguments object for `%s`, with no prose.", step.Tool)
}

func buildToolExecutionCorrection(step ToonStep, issues []string, retryNumber int) string {
	var sb strings.Builder
	sb.WriteString("HARD TOOL RETRY.\n")
	sb.WriteString(fmt.Sprintf("Retry=%d.\n", retryNumber))
	sb.WriteString("Return exactly one markdown block only.\n")
	sb.WriteString("No plain JSON. No prose. No text before or after the block.\n")
	sb.WriteString("Return only the arguments object for tool `" + step.Tool + "`.\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	sb.WriteString(fmt.Sprintf("Current required step is still: step %d: %s\n", step.Number, step.Tool))
	sb.WriteString("Return only:\n")
	sb.WriteString("```\n{...arguments...}\n```")
	return strings.TrimSpace(sb.String())
}

func buildFinalExecutionCorrection(issues []string, retryNumber int) string {
	var sb strings.Builder
	sb.WriteString("HARD FINAL RETRY.\n")
	sb.WriteString(fmt.Sprintf("Retry=%d.\n", retryNumber))
	sb.WriteString("Answer again as direct plain text only.\n")
	sb.WriteString("No JSON. No fences. Keep following the execution-layer system prompt and the existing conversation messages.\n")
	for _, issue := range issues {
		sb.WriteString("- " + issue + "\n")
	}
	return strings.TrimSpace(sb.String())
}

func validateExecutorToolReply(reply string, step ToonStep, registry map[string]ToolSpec) (map[string]interface{}, []string) {
	blockContent, ok := unwrapFullMarkdownFence(reply)
	var issues []string
	if !ok {
		return nil, []string{"tool step must be wrapped in exactly one markdown block"}
	}

	trimmed := strings.TrimSpace(blockContent)
	if trimmed == "" {
		return nil, []string{"tool step returned empty output"}
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
	if _, exists := raw["action"]; exists {
		issues = append(issues, "do not return `action`; return only the arguments object")
	}
	if _, exists := raw["name"]; exists {
		issues = append(issues, "do not return `name`; the tool is already fixed by the plan")
	}
	if _, exists := raw["arguments"]; exists {
		issues = append(issues, "do not return `arguments` wrapper; the whole object must be the arguments object")
	}

	parsed := map[string]interface{}{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		issues = append(issues, "tool step JSON must be a plain object of arguments")
		return nil, issues
	}
	if parsed == nil {
		issues = append(issues, "arguments object must not be null")
	}
	spec, ok := registry[step.Tool]
	if !ok {
		issues = append(issues, fmt.Sprintf("tool `%s` is not present in the registry", step.Tool))
	}
	if ok {
		for _, required := range spec.Required {
			if _, exists := parsed[required]; !exists {
				issues = append(issues, fmt.Sprintf("missing required argument `%s`", required))
			}
		}
	}
	if len(issues) > 0 {
		return nil, issues
	}
	return parsed, nil
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

	var messages []Message
	var toolStep *ToonStep
	if len(plan.Steps) > 0 {
		step := plan.Steps[0]
		toolStep = &step
		payloadToon, err := renderExecutorPayloadToon(buildExecutorPayload(req, plan, registry))
		if err != nil {
			return ChatResponse{}, fmt.Errorf("executor toon payload error: %w", err)
		}
		messages = []Message{
			{Role: "system", Content: contentFromString(buildToolExecutionPrompt(step, registry))},
			{Role: "assistant", Content: contentFromString(toolExecutionPrimerMessage(step))},
			{Role: "user", Content: contentFromString(payloadToon)},
		}
	} else {
		messages = append(
			[]Message{{Role: "system", Content: contentFromString(buildFinalExecutionPrompt())}},
			buildFinalExecutionMessages(req)...,
		)
	}

	var lastIssues []string
	var lastReply string
	for attempt := 1; attempt <= cfg.ExecutorMaxRetries; attempt++ {
		var reply string
		var err error
		if toolStep != nil {
			reply, err = callUpstreamTextUntil(model, messages, func(reply string) streamStopDecision {
				_, issues := validateExecutorToolReply(reply, *toolStep, registry)
				if len(issues) == 0 {
					return streamStopDecision{
						Stop:       true,
						LogIcon:    "⚡",
						LogMessage: "executor stream already contains a complete JSON arguments block; stopping upstream stream early",
					}
				}
				return streamStopDecision{}
			})
		} else {
			reply, err = callUpstreamText(model, messages)
		}
		if err != nil {
			return ChatResponse{}, fmt.Errorf("executor upstream error: %w", err)
		}
		lastReply = reply
		logSection(fmt.Sprintf("EXECUTOR ATTEMPT %d/%d", attempt, cfg.ExecutorMaxRetries))
		fmt.Printf("│\n%s\n", prefixLines(reply, "│  "))

		if toolStep != nil {
			arguments, issues := validateExecutorToolReply(reply, *toolStep, registry)
			if len(issues) == 0 {
				argsJSON, _ := json.Marshal(arguments)
				return makeOpenAIResponse("", model, []ToolCall{{
					ID:       "call_" + uuid.New().String()[:8],
					Type:     "function",
					Function: FunctionCall{Name: toolStep.Tool, Arguments: string(argsJSON)},
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

	plan, err := requestValidatedPlan(req, registry)
	if err != nil {
		return makeOpenAIResponse("Planner error: "+err.Error(), model, nil)
	}

	logSection("VALIDATED NEXT ACTION")
	fmt.Printf("│\n%s\n", prefixLines(renderToonPlan(plan), "│  "))
	if len(plan.Steps) == 0 && strings.TrimSpace(plan.Final) != "" && !hasToolActivityAfterLatestUser(req.Messages) {
		logResult("⚡", "planner returned a direct final answer without tool usage; skipping executor")
		return makeOpenAIResponse(plan.Final, model, nil)
	}

	result, err := runExecutor(req, plan, registry)
	if err != nil {
		return makeOpenAIResponse("Execution error: "+err.Error(), model, nil)
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
	fmt.Printf("  Max req bytes:  %d\n", cfg.MaxRequestBytes)
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
