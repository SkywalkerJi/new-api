package aws

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
)

type AwsClaudeRequest struct {
	// AnthropicVersion should be "bedrock-2023-05-31"
	AnthropicVersion string              `json:"anthropic_version"`
	AnthropicBeta    json.RawMessage     `json:"anthropic_beta,omitempty"`
	System           any                 `json:"system,omitempty"`
	Messages         []dto.ClaudeMessage `json:"messages"`
	MaxTokens        uint                `json:"max_tokens,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             float64             `json:"top_p,omitempty"`
	TopK             int                 `json:"top_k,omitempty"`
	StopSequences    []string            `json:"stop_sequences,omitempty"`
	Tools            any                 `json:"tools,omitempty"`
	ToolChoice       any                 `json:"tool_choice,omitempty"`
	Thinking         *dto.Thinking       `json:"thinking,omitempty"`
	OutputConfig     json.RawMessage     `json:"output_config,omitempty"`
	//Metadata         json.RawMessage     `json:"metadata,omitempty"`
}

func formatRequest(requestBody io.Reader, requestHeader http.Header) (*AwsClaudeRequest, error) {
	var awsClaudeRequest AwsClaudeRequest
	err := common.DecodeJson(requestBody, &awsClaudeRequest)
	if err != nil {
		return nil, err
	}
	awsClaudeRequest.AnthropicVersion = "bedrock-2023-05-31"

	// check header anthropic-beta
	anthropicBetaValues := requestHeader.Get("anthropic-beta")
	if len(anthropicBetaValues) > 0 {
		var tempArray []string
		tempArray = strings.Split(anthropicBetaValues, ",")
		if len(tempArray) > 0 {
			betaJson, err := json.Marshal(tempArray)
			if err != nil {
				return nil, err
			}
			awsClaudeRequest.AnthropicBeta = betaJson
		}
	}
	logger.LogJson(context.Background(), "json", awsClaudeRequest)
	return &awsClaudeRequest, nil
}

// NovaMessage Nova模型使用messages-v1格式
type NovaMessage struct {
	Role    string        `json:"role"`
	Content []NovaContent `json:"content"`
}

type NovaContent struct {
	Text string `json:"text"`
}

type NovaRequest struct {
	SchemaVersion   string               `json:"schemaVersion"`             // 请求版本，例如 "1.0"
	Messages        []NovaMessage        `json:"messages"`                  // 对话消息列表
	InferenceConfig *NovaInferenceConfig `json:"inferenceConfig,omitempty"` // 推理配置，可选
}

type NovaInferenceConfig struct {
	MaxTokens     int      `json:"maxTokens,omitempty"`     // 最大生成的 token 数
	Temperature   float64  `json:"temperature,omitempty"`   // 随机性 (默认 0.7, 范围 0-1)
	TopP          float64  `json:"topP,omitempty"`          // nucleus sampling (默认 0.9, 范围 0-1)
	TopK          int      `json:"topK,omitempty"`          // 限制候选 token 数 (默认 50, 范围 0-128)
	StopSequences []string `json:"stopSequences,omitempty"` // 停止生成的序列
}

// 转换OpenAI请求为Nova格式
func convertToNovaRequest(req *dto.GeneralOpenAIRequest) *NovaRequest {
	novaMessages := make([]NovaMessage, len(req.Messages))
	for i, msg := range req.Messages {
		novaMessages[i] = NovaMessage{
			Role:    msg.Role,
			Content: []NovaContent{{Text: msg.StringContent()}},
		}
	}

	novaReq := &NovaRequest{
		SchemaVersion: "messages-v1",
		Messages:      novaMessages,
	}

	// 设置推理配置
	if (req.MaxTokens != nil && *req.MaxTokens != 0) || (req.Temperature != nil && *req.Temperature != 0) || (req.TopP != nil && *req.TopP != 0) || (req.TopK != nil && *req.TopK != 0) || req.Stop != nil {
		novaReq.InferenceConfig = &NovaInferenceConfig{}
		if req.MaxTokens != nil && *req.MaxTokens != 0 {
			novaReq.InferenceConfig.MaxTokens = int(*req.MaxTokens)
		}
		if req.Temperature != nil && *req.Temperature != 0 {
			novaReq.InferenceConfig.Temperature = *req.Temperature
		}
		if req.TopP != nil && *req.TopP != 0 {
			novaReq.InferenceConfig.TopP = *req.TopP
		}
		if req.TopK != nil && *req.TopK != 0 {
			novaReq.InferenceConfig.TopK = *req.TopK
		}
		if req.Stop != nil {
			if stopSequences := parseStopSequences(req.Stop); len(stopSequences) > 0 {
				novaReq.InferenceConfig.StopSequences = stopSequences
			}
		}
	}

	return novaReq
}

// parseStopSequences 解析停止序列，支持字符串或字符串数组
func parseStopSequences(stop any) []string {
	if stop == nil {
		return nil
	}

	switch v := stop.(type) {
	case string:
		if v != "" {
			return []string{v}
		}
	case []string:
		return v
	case []interface{}:
		var sequences []string
		for _, item := range v {
			if str, ok := item.(string); ok && str != "" {
				sequences = append(sequences, str)
			}
		}
		return sequences
	}
	return nil
}

// -------- Z.AI GLM (OpenAI-Compatible on Bedrock) --------

// GlmMessage aligns with OpenAI chat.completions message.
// Content uses json.RawMessage to pass through strings or multimodal arrays verbatim.
type GlmMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCallId string          `json:"tool_call_id,omitempty"`
	ToolCalls  any             `json:"tool_calls,omitempty"`
}

// GlmRequest is the body sent to Bedrock zai.glm-* endpoints.
// Optional scalar fields use pointer + omitempty (project Rule 6) so explicit zero/false
// values sent by the client are preserved upstream.
type GlmRequest struct {
	Messages    []GlmMessage `json:"messages"`
	MaxTokens   *int         `json:"max_tokens,omitempty"`
	Temperature *float64     `json:"temperature,omitempty"`
	TopP        *float64     `json:"top_p,omitempty"`
	Stream      *bool        `json:"stream,omitempty"`
	Tools       any          `json:"tools,omitempty"`
	ToolChoice  any          `json:"tool_choice,omitempty"`
	Thinking    string       `json:"thinking,omitempty"` // "enabled" | "disabled"
	Stop        any          `json:"stop,omitempty"`
}

// convertToGlmRequest converts internal OpenAI-format request to GLM / Bedrock Chat Completions.
// Client did not send a field → leave destination nil → omitted on marshal.
func convertToGlmRequest(req *dto.GeneralOpenAIRequest) *GlmRequest {
	msgs := make([]GlmMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		gm := GlmMessage{
			Role:       m.Role,
			ToolCallId: m.ToolCallId,
		}
		if m.Content != nil {
			if raw, ok := m.Content.(json.RawMessage); ok {
				// Already raw — pass through verbatim.
				if len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
					gm.Content = raw
				}
			} else if raw, err := common.Marshal(m.Content); err == nil {
				if !bytes.Equal(raw, []byte("null")) {
					gm.Content = raw
				}
			} else {
				common.SysError(fmt.Sprintf("aws glm convert: marshal message content failed: %v", err))
			}
		}
		if m.Name != nil {
			gm.Name = *m.Name
		}
		if len(m.ToolCalls) > 0 {
			gm.ToolCalls = m.ToolCalls
		}
		msgs = append(msgs, gm)
	}
	g := &GlmRequest{Messages: msgs}
	if req.MaxTokens != nil {
		v := int(*req.MaxTokens)
		g.MaxTokens = &v
	}
	if req.Temperature != nil {
		v := *req.Temperature
		g.Temperature = &v
	}
	if req.TopP != nil {
		v := *req.TopP
		g.TopP = &v
	}
	if req.Stream != nil && *req.Stream {
		v := true
		g.Stream = &v
	}
	if len(req.Tools) > 0 {
		g.Tools = req.Tools
	}
	if req.ToolChoice != nil {
		g.ToolChoice = req.ToolChoice
	}
	if req.Stop != nil {
		g.Stop = req.Stop
	}
	return g
}

// -------- DeepSeek V3.2 (OpenAI-Compatible on Bedrock) --------

// DeepSeekMessage 是 DeepSeek V3.2 请求体里的 message。
// Content 使用 json.RawMessage + omitempty：client 若已是 raw 则原样透传，
// nil 则在 marshal 时被丢弃（避免输出 "content": null）。
type DeepSeekMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCallId string          `json:"tool_call_id,omitempty"`
	ToolCalls  any             `json:"tool_calls,omitempty"`
}

// DeepSeekRequest is the body sent to Bedrock deepseek.* endpoints.
// Schema follows OpenAI Chat Completions (verified against AWS DeepSeek V3.2
// model-card as of 2026-04-16). Optional scalar fields use pointer +
// omitempty (project Rule 6) so explicit zero/false values from the client
// are preserved upstream.
type DeepSeekRequest struct {
	Messages         []DeepSeekMessage `json:"messages"`
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	Stream           *bool             `json:"stream,omitempty"`
	StreamOptions    any               `json:"stream_options,omitempty"`
	Tools            any               `json:"tools,omitempty"`
	ToolChoice       any               `json:"tool_choice,omitempty"`
	Stop             any               `json:"stop,omitempty"`
	FrequencyPenalty *float64          `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64          `json:"presence_penalty,omitempty"`
	ResponseFormat   any               `json:"response_format,omitempty"`
}

// convertToDeepSeekRequest 把内部 OpenAI 请求转成 DeepSeek / Bedrock
// Chat Completions 格式。client 没发的字段 → 目标保持 nil → marshal 时丢弃。
func convertToDeepSeekRequest(req *dto.GeneralOpenAIRequest) *DeepSeekRequest {
	msgs := make([]DeepSeekMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		dm := DeepSeekMessage{
			Role:       m.Role,
			ToolCallId: m.ToolCallId,
		}
		if m.Content != nil {
			if raw, ok := m.Content.(json.RawMessage); ok {
				if len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
					dm.Content = raw
				}
			} else if raw, err := common.Marshal(m.Content); err == nil {
				if !bytes.Equal(raw, []byte("null")) {
					dm.Content = raw
				}
			} else {
				common.SysError(fmt.Sprintf("aws deepseek convert: marshal message content failed: %v", err))
			}
		}
		if m.Name != nil {
			dm.Name = *m.Name
		}
		if len(m.ToolCalls) > 0 {
			dm.ToolCalls = m.ToolCalls
		}
		msgs = append(msgs, dm)
	}
	d := &DeepSeekRequest{Messages: msgs}
	if req.MaxTokens != nil {
		v := int(*req.MaxTokens)
		d.MaxTokens = &v
	}
	if req.Temperature != nil {
		v := *req.Temperature
		d.Temperature = &v
	}
	if req.TopP != nil {
		v := *req.TopP
		d.TopP = &v
	}
	if req.Stream != nil && *req.Stream {
		v := true
		d.Stream = &v
	}
	if req.StreamOptions != nil {
		d.StreamOptions = req.StreamOptions
	}
	if len(req.Tools) > 0 {
		d.Tools = req.Tools
	}
	if req.ToolChoice != nil {
		d.ToolChoice = req.ToolChoice
	}
	if req.Stop != nil {
		d.Stop = req.Stop
	}
	if req.FrequencyPenalty != nil {
		v := *req.FrequencyPenalty
		d.FrequencyPenalty = &v
	}
	if req.PresencePenalty != nil {
		v := *req.PresencePenalty
		d.PresencePenalty = &v
	}
	if req.ResponseFormat != nil {
		d.ResponseFormat = req.ResponseFormat
	}
	return d
}
