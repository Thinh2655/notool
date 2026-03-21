package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

func loadEnvFile(filename string) {
	f, err := os.Open(filename)
	if err != nil {
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
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			if os.Getenv(key) == "" {
				os.Setenv(key, value)
			}
		}
	}
}

func testSize(url, apiKey, model string, charCount int) (int, error) {
	// ~4 chars per token
	content := strings.Repeat("hello ", charCount/6)

	body, _ := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": content},
		},
		"stream":     false,
		"max_tokens": 1,
	})

	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	io.ReadAll(resp.Body)
	return resp.StatusCode, nil
}

func main() {
	loadEnvFile(".env")
	url := os.Getenv("UPSTREAM_URL")
	apiKey := os.Getenv("UPSTREAM_API_KEY")
	model := os.Getenv("UPSTREAM_MODEL")

	if url == "" || apiKey == "" || model == "" {
		fmt.Println("Missing UPSTREAM_URL, UPSTREAM_API_KEY, or UPSTREAM_MODEL in .env")
		return
	}

	fmt.Printf("Probing upstream: %s (model: %s)\n", url, model)
	fmt.Println("Binary search for max token limit...\n")

	// Binary search: low = OK, high = TOO_LARGE
	low := 1000    // 1K chars (~250 tokens) - should work
	high := 800000 // 800K chars (~200K tokens) - likely fails

	// Verify low works
	fmt.Printf("Testing baseline %d chars (~%d tokens)... ", low, low/4)
	status, err := testSize(url, apiKey, model, low)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("HTTP %d\n", status)
	if status == 413 {
		fmt.Println("Even baseline failed! Limit is very small.")
		return
	}

	// Verify high fails
	fmt.Printf("Testing upper bound %d chars (~%d tokens)... ", high, high/4)
	status, err = testSize(url, apiKey, model, high)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("HTTP %d\n", status)
	if status != 413 {
		fmt.Printf("Upper bound passed! Limit is > %d tokens\n", high/4)
		return
	}

	fmt.Println()

	// Binary search
	for high-low > 2000 {
		mid := (low + high) / 2
		fmt.Printf("Testing %d chars (~%d tokens)... ", mid, mid/4)
		status, err := testSize(url, apiKey, model, mid)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}
		fmt.Printf("HTTP %d\n", status)
		if status == 413 {
			high = mid
		} else {
			low = mid
		}
	}

	estimatedTokens := low / 4
	fmt.Printf("\n========================================\n")
	fmt.Printf("Max input: ~%d chars (~%d tokens)\n", low, estimatedTokens)
	fmt.Printf("Suggested MAX_TOKENS = %d\n", estimatedTokens-2000)
	fmt.Printf("========================================\n")
}
