package aws

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/types"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestDoAwsClientRequest_AppliesRuntimeHeaderOverrideToAnthropicBeta(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	info := &relaycommon.RelayInfo{
		OriginModelName:           "claude-3-5-sonnet-20240620",
		IsStream:                  false,
		UseRuntimeHeadersOverride: true,
		RuntimeHeadersOverride: map[string]any{
			"anthropic-beta": "computer-use-2025-01-24",
		},
		ChannelMeta: &relaycommon.ChannelMeta{
			ApiKey:            "access-key|secret-key|us-east-1",
			UpstreamModelName: "claude-3-5-sonnet-20240620",
		},
	}

	requestBody := bytes.NewBufferString(`{"messages":[{"role":"user","content":"hello"}],"max_tokens":128}`)
	adaptor := &Adaptor{}

	_, err := doAwsClientRequest(ctx, info, adaptor, requestBody)
	require.NoError(t, err)

	awsReq, ok := adaptor.AwsReq.(*bedrockruntime.InvokeModelInput)
	require.True(t, ok)

	var payload map[string]any
	require.NoError(t, common.Unmarshal(awsReq.Body, &payload))

	anthropicBeta, exists := payload["anthropic_beta"]
	require.True(t, exists)

	values, ok := anthropicBeta.([]any)
	require.True(t, ok)
	require.Equal(t, []any{"computer-use-2025-01-24"}, values)
}

func TestBuildClaudeNativeBody_AddsAnthropicVersionAndBeta(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	ctx.Request.Header.Set("anthropic-beta", "computer-use-2025-01-24,prompt-caching-2024-07-31")

	info := &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{}}
	reqBody := bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}`)

	body, err := buildClaudeNativeBody(ctx, info, reqBody, ctx.Request.Header)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, common.Unmarshal(body, &payload))
	require.Equal(t, "bedrock-2023-05-31", payload["anthropic_version"])

	beta, ok := payload["anthropic_beta"].([]any)
	require.True(t, ok)
	require.Equal(t, []any{"computer-use-2025-01-24", "prompt-caching-2024-07-31"}, beta)
}

func TestResolveAwsModelId(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		model  string
		region string
		want   string
	}{
		{"opus-4-6 强制 global 前缀", "claude-opus-4-6", "us-east-1", "global.anthropic.claude-opus-4-6-v1"},
		{"opus-4-6 从东京也是 global", "claude-opus-4-6", "ap-northeast-1", "global.anthropic.claude-opus-4-6-v1"},
		{"sonnet-4-6 东京使用 jp 前缀", "claude-sonnet-4-6", "ap-northeast-1", "jp.anthropic.claude-sonnet-4-6"},
		{"sonnet-4-5 美东使用 us 前缀", "claude-sonnet-4-5-20250929", "us-east-1", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{"3-5-sonnet ap-southeast-1 使用 apac", "claude-3-5-sonnet-20240620", "ap-southeast-1", "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"},
		{"未知 region 不加前缀", "claude-3-5-sonnet-20240620", "me-central-1", "anthropic.claude-3-5-sonnet-20240620-v1:0"},
		{"非内置模型原样返回", "custom-model", "us-east-1", "custom-model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveAwsModelId(tc.model, tc.region)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestGetRequestURL_ApiKeyMode(t *testing.T) {
	t.Parallel()

	mkInfo := func(model, key string, stream bool) *relaycommon.RelayInfo {
		return &relaycommon.RelayInfo{
			IsStream: stream,
			ChannelMeta: &relaycommon.ChannelMeta{
				ApiKey:            key,
				UpstreamModelName: model,
				ChannelOtherSettings: dto.ChannelOtherSettings{
					AwsKeyType: dto.AwsKeyTypeApiKey,
				},
			},
		}
	}

	cases := []struct {
		name    string
		info    *relaycommon.RelayInfo
		wantURL string
		wantErr bool
	}{
		{
			name:    "sonnet-4-6 东京非流式 -> jp 前缀 + invoke",
			info:    mkInfo("claude-sonnet-4-6", "bedrock-key-xxx|ap-northeast-1", false),
			wantURL: "https://bedrock-runtime.ap-northeast-1.amazonaws.com/model/jp.anthropic.claude-sonnet-4-6/invoke",
		},
		{
			name:    "opus-4-6 东京流式 -> global 前缀 + invoke-with-response-stream",
			info:    mkInfo("claude-opus-4-6", "bedrock-key-xxx|ap-northeast-1", true),
			wantURL: "https://bedrock-runtime.ap-northeast-1.amazonaws.com/model/global.anthropic.claude-opus-4-6-v1/invoke-with-response-stream",
		},
		{
			name:    "3-5-sonnet 美东非流式 -> us 前缀 + invoke",
			info:    mkInfo("claude-3-5-sonnet-20240620", "bedrock-key-xxx|us-east-1", false),
			wantURL: "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
		},
		{
			name:    "密钥格式错误 -> 报错",
			info:    mkInfo("claude-3-5-sonnet-20240620", "bedrock-key-xxx", false),
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := &Adaptor{}
			url, err := a.GetRequestURL(tc.info)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantURL, url)
			require.Equal(t, ClientModeApiKey, a.ClientMode)
			require.NotEmpty(t, a.BearerToken)
			require.NotContains(t, a.BearerToken, "|")
		})
	}
}

func TestSetupRequestHeader_ApiKeyMode_StripsRegionFromToken(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{
			ApiKey: "bedrock-bearer-token-xxx|ap-northeast-1",
		},
	}

	a := &Adaptor{
		ClientMode:  ClientModeApiKey,
		BearerToken: "bedrock-bearer-token-xxx",
	}
	header := http.Header{}
	err := a.SetupRequestHeader(ctx, &header, info)
	require.NoError(t, err)
	require.Equal(t, "Bearer bedrock-bearer-token-xxx", header.Get("Authorization"))
}

func TestSetupRequestHeader_AkskMode_NoBearerToken(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{
			ApiKey: "ak|sk|us-east-1",
		},
	}

	a := &Adaptor{ClientMode: ClientModeAKSK}
	header := http.Header{}
	err := a.SetupRequestHeader(ctx, &header, info)
	require.NoError(t, err)
	require.Empty(t, header.Get("Authorization"))
}

func TestIsGlmModel(t *testing.T) {
	t.Parallel()
	cases := map[string]bool{
		"glm-5":           true,
		"glm-4.7":         true,
		"zai.glm-5":       true,
		"claude-opus-4-6": false,
		"nova-pro-v1:0":   false,
		"":                false,
	}
	for in, want := range cases {
		require.Equalf(t, want, isGlmModel(in), "isGlmModel(%q)", in)
	}
}

func TestResolveAwsModelId_GlmNoCrossRegionPrefix(t *testing.T) {
	t.Parallel()
	// GLM does not support geo/global inference profiles; every region should return the raw zai.glm-*
	require.Equal(t, "zai.glm-5", resolveAwsModelId("glm-5", "us-east-1"))
	require.Equal(t, "zai.glm-5", resolveAwsModelId("glm-5", "eu-west-2"))
	require.Equal(t, "zai.glm-5", resolveAwsModelId("glm-5", "ap-northeast-1"))
	require.Equal(t, "zai.glm-4.7", resolveAwsModelId("glm-4.7", "us-west-2"))
}

func TestConvertToGlmRequest_PreservesOptionalFields(t *testing.T) {
	t.Parallel()
	zero := 0.0
	maxTokens := uint(256)
	streamTrue := true

	req := &dto.GeneralOpenAIRequest{
		Model: "glm-5",
		Messages: []dto.Message{
			{Role: "user", Content: "hi"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &zero, // explicit 0 — MUST be emitted
		Stream:      &streamTrue,
	}

	glm := convertToGlmRequest(req)
	require.NotNil(t, glm)
	require.Len(t, glm.Messages, 1)
	require.Equal(t, "user", glm.Messages[0].Role)

	require.NotNil(t, glm.MaxTokens)
	require.Equal(t, 256, *glm.MaxTokens)

	require.NotNil(t, glm.Temperature)
	require.Equal(t, 0.0, *glm.Temperature)

	require.NotNil(t, glm.Stream)
	require.True(t, *glm.Stream)

	require.Nil(t, glm.TopP, "TopP not provided by caller → must stay nil (omitted on marshal)")

	// Verify anthropic_version is NOT in the marshaled output
	raw, err := common.Marshal(glm)
	require.NoError(t, err)
	require.NotContains(t, string(raw), "anthropic_version")
	require.NotContains(t, string(raw), "system")
}

func TestConvertToGlmRequest_ContentRawMessagePassthrough(t *testing.T) {
	t.Parallel()
	req := &dto.GeneralOpenAIRequest{
		Model: "glm-5",
		Messages: []dto.Message{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
	}
	glm := convertToGlmRequest(req)
	require.Len(t, glm.Messages, 1)
	// Must NOT be re-encoded to "\"hi\""
	require.Equal(t, `"hi"`, string(glm.Messages[0].Content))
}

func TestConvertToGlmRequest_NilContentDropsField(t *testing.T) {
	t.Parallel()
	req := &dto.GeneralOpenAIRequest{
		Model: "glm-5",
		Messages: []dto.Message{
			{Role: "assistant", Content: nil},
		},
	}
	glm := convertToGlmRequest(req)
	require.Len(t, glm.Messages, 1)
	require.Empty(t, glm.Messages[0].Content, "nil content must leave GlmMessage.Content empty so omitempty drops it")

	raw, err := common.Marshal(glm)
	require.NoError(t, err)
	require.NotContains(t, string(raw), `"content":null`, "must never emit literal null content")
}

func TestConvertOpenAIRequest_RoutesGlm(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{UpstreamModelName: "glm-5"},
	}
	a := &Adaptor{}
	req := &dto.GeneralOpenAIRequest{
		Model: "glm-5",
		Messages: []dto.Message{
			{Role: "user", Content: "hi"},
		},
	}

	out, err := a.ConvertOpenAIRequest(ctx, info, req)
	require.NoError(t, err)

	glm, ok := out.(*GlmRequest)
	require.True(t, ok, "GLM request should be routed to GlmRequest, got %T", out)
	require.True(t, a.IsGlm)
	require.False(t, a.IsNova)
	require.Len(t, glm.Messages, 1)
	require.Equal(t, "user", glm.Messages[0].Role)
	require.Contains(t, string(glm.Messages[0].Content), "hi")
}

func TestDoAwsClientRequest_Glm_BuildsGlmBodyWithoutAnthropicVersion(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	info := &relaycommon.RelayInfo{
		OriginModelName: "glm-5",
		IsStream:        false,
		ChannelMeta: &relaycommon.ChannelMeta{
			ApiKey:            "ak|sk|us-east-1",
			UpstreamModelName: "glm-5",
		},
	}

	glmBody := bytes.NewBufferString(`{"messages":[{"role":"user","content":"hello"}],"max_tokens":128}`)
	a := &Adaptor{IsGlm: true}

	_, err := doAwsClientRequest(ctx, info, a, glmBody)
	require.NoError(t, err)

	awsReq, ok := a.AwsReq.(*bedrockruntime.InvokeModelInput)
	require.True(t, ok)
	require.Equal(t, "zai.glm-5", *awsReq.ModelId)

	var payload map[string]any
	require.NoError(t, common.Unmarshal(awsReq.Body, &payload))
	_, hasVersion := payload["anthropic_version"]
	require.False(t, hasVersion, "GLM body must NOT carry anthropic_version")
	require.NotNil(t, payload["messages"])

	// Lock content passthrough — body must contain the exact messages/max_tokens caller sent.
	msgs, ok := payload["messages"].([]any)
	require.True(t, ok)
	require.Len(t, msgs, 1)
	first, ok := msgs[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", first["role"])
	require.Equal(t, "hello", first["content"])
	require.Equal(t, float64(128), payload["max_tokens"])
}

func TestDoAwsClientRequest_Glm_Stream_UsesStreamInput(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	info := &relaycommon.RelayInfo{
		OriginModelName: "glm-5",
		IsStream:        true,
		ChannelMeta: &relaycommon.ChannelMeta{
			ApiKey:            "ak|sk|us-east-1",
			UpstreamModelName: "glm-5",
		},
	}
	body := bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}]}`)
	a := &Adaptor{IsGlm: true}

	_, err := doAwsClientRequest(ctx, info, a, body)
	require.NoError(t, err)

	_, ok := a.AwsReq.(*bedrockruntime.InvokeModelWithResponseStreamInput)
	require.True(t, ok, "stream mode must use InvokeModelWithResponseStreamInput, got %T", a.AwsReq)

	streamReq := a.AwsReq.(*bedrockruntime.InvokeModelWithResponseStreamInput)
	require.Equal(t, "zai.glm-5", *streamReq.ModelId)
	require.Equal(t, "application/json", *streamReq.ContentType)
	require.Equal(t, "application/json", *streamReq.Accept)

	var payload map[string]any
	require.NoError(t, common.Unmarshal(streamReq.Body, &payload))
	require.NotNil(t, payload["messages"])
}

func TestAwsGlmHandler_WritesOpenAIResponse(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{UpstreamModelName: "glm-5"},
		RelayFormat: types.RelayFormatOpenAI,
	}

	raw := []byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)

	usage, apiErr := decodeGlmNonStreamBody(ctx, info, raw)
	require.Nil(t, apiErr)
	require.NotNil(t, usage)
	require.Equal(t, 3, usage.PromptTokens)
	require.Equal(t, 1, usage.CompletionTokens)
	require.Equal(t, 4, usage.TotalTokens)
	require.Contains(t, recorder.Body.String(), `"content":"hi"`)
}

func TestDecodeGlmNonStreamBody_PreservesUpstreamFields(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{UpstreamModelName: "glm-5"},
		RelayFormat: types.RelayFormatOpenAI,
	}

	raw := []byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi","reasoning_content":"thought A","thinking":"block B"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)

	usage, apiErr := decodeGlmNonStreamBody(ctx, info, raw)
	require.Nil(t, apiErr)
	require.NotNil(t, usage)
	require.Equal(t, 3, usage.PromptTokens)

	body := recorder.Body.String()
	require.Contains(t, body, `"reasoning_content":"thought A"`, "upstream reasoning_content must be forwarded verbatim")
	require.Contains(t, body, `"thinking":"block B"`, "upstream thinking must be forwarded verbatim")
	require.Contains(t, body, `"content":"hi"`)
}

func TestBuildApiKeyRequestBody_Glm_PassesThroughRaw(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	info := &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{}}
	a := &Adaptor{IsGlm: true}
	original := []byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`)

	out, err := a.buildApiKeyRequestBody(ctx, info, bytes.NewReader(original))
	require.NoError(t, err)
	require.Equal(t, string(original), string(out))
	require.NotContains(t, string(out), "anthropic_version")
}

func TestBuildApiKeyRequestBody_Claude_WrapsWithAnthropicVersion(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	info := &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{}}
	a := &Adaptor{IsGlm: false} // Claude default
	original := []byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`)

	out, err := a.buildApiKeyRequestBody(ctx, info, bytes.NewReader(original))
	require.NoError(t, err)
	require.Contains(t, string(out), `"anthropic_version":"bedrock-2023-05-31"`)
}

func TestDoResponse_ApiKey_Glm_UsesOpenAIHandler(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	responseBody := `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","thinking":"T"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`

	// Build a fake http.Response
	resp := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader([]byte(responseBody))),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	info := &relaycommon.RelayInfo{
		IsStream:    false,
		RelayFormat: types.RelayFormatOpenAI,
		ChannelMeta: &relaycommon.ChannelMeta{UpstreamModelName: "glm-5"},
	}
	a := &Adaptor{ClientMode: ClientModeApiKey, IsGlm: true}

	usage, apiErr := a.DoResponse(ctx, resp, info)
	require.Nil(t, apiErr)
	require.NotNil(t, usage, "OpenAI handler must return a non-nil usage")
	u, ok := usage.(*dto.Usage)
	require.True(t, ok, "usage must be *dto.Usage, got %T", usage)
	require.Equal(t, 7, u.TotalTokens)

	body := recorder.Body.String()
	require.Contains(t, body, `"content":"ok"`)
	require.Contains(t, body, `"thinking":"T"`, "thinking must pass through via OpenAI handler")
}

func TestDoResponse_ApiKey_Glm_Stream_UsesOaiStreamHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	// StreamScannerHandler needs a positive streaming timeout; in normal startup
	// this is set by common.init, but unit tests must prime it themselves.
	oldStreamingTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldStreamingTimeout
	})

	// Simulated OpenAI-compatible SSE stream from Bedrock (API Key mode)
	sseBody := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2,\"total_tokens\":4}}\n\n" +
		"data: [DONE]\n\n"

	resp := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader([]byte(sseBody))),
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
	}

	info := &relaycommon.RelayInfo{
		IsStream:    true,
		RelayFormat: types.RelayFormatOpenAI,
		ChannelMeta: &relaycommon.ChannelMeta{UpstreamModelName: "glm-5"},
	}
	a := &Adaptor{ClientMode: ClientModeApiKey, IsGlm: true}

	usage, apiErr := a.DoResponse(ctx, resp, info)
	require.Nil(t, apiErr)
	require.NotNil(t, usage)

	out := recorder.Body.String()
	// Chunks should be forwarded to the client as SSE; at minimum both deltas arrive.
	require.Contains(t, out, `"content":"hel"`)
	require.Contains(t, out, `"content":"lo"`)
}

func TestIsDeepSeekModel(t *testing.T) {
	t.Parallel()
	cases := map[string]bool{
		"deepseek.v3.2":       true,
		"deepseek.r1-v1:0":    true, // prefix 匹配；实际路由仍只有 v3.2 有 map 条目
		"glm-5":               false,
		"zai.glm-5":           false,
		"claude-opus-4-6":     false,
		"anthropic.claude-x":  false,
		"nova-pro-v1:0":       false,
		"":                    false,
	}
	for in, want := range cases {
		require.Equalf(t, want, isDeepSeekModel(in), "isDeepSeekModel(%q)", in)
	}
}

func TestResolveAwsModelId_DeepSeek_NoCrossRegionPrefix(t *testing.T) {
	t.Parallel()
	// DeepSeek V3.2 是 serverless 模型，所有官方支持区域都必须返回 bare ID。
	// 回归护栏：Bedrock 会用 validation error 拒绝 us.deepseek.v3.2 等前缀 ID，
	// 见 https://github.com/kirodotdev/Kiro/issues/6711。
	regions := []string{
		"us-east-1", "us-east-2", "us-west-2",
		"eu-north-1", "eu-west-2",
		"ap-northeast-1", // fine-grained map 有 "jp" 前缀，必须不触发
		"ap-south-1", "ap-southeast-2", "ap-southeast-3",
		"sa-east-1",
	}
	for _, region := range regions {
		require.Equalf(t, "deepseek.v3.2",
			resolveAwsModelId("deepseek.v3.2", region),
			"region=%s", region)
	}
}

func TestConvertToDeepSeekRequest_PreservesOptionalFields(t *testing.T) {
	t.Parallel()
	zero := 0.0
	maxTokens := uint(256)
	streamTrue := true

	req := &dto.GeneralOpenAIRequest{
		Model: "deepseek.v3.2",
		Messages: []dto.Message{
			{Role: "user", Content: "hi"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &zero, // 显式 0 —— 必须被透传（Rule 6）
		Stream:      &streamTrue,
	}

	ds := convertToDeepSeekRequest(req)
	require.NotNil(t, ds)
	require.Len(t, ds.Messages, 1)
	require.Equal(t, "user", ds.Messages[0].Role)

	require.NotNil(t, ds.MaxTokens)
	require.Equal(t, 256, *ds.MaxTokens)

	require.NotNil(t, ds.Temperature)
	require.Equal(t, 0.0, *ds.Temperature)

	require.NotNil(t, ds.Stream)
	require.True(t, *ds.Stream)

	require.Nil(t, ds.TopP, "TopP 未提供 → 必须保持 nil（marshal 时被 omitempty 丢弃）")

	raw, err := common.Marshal(ds)
	require.NoError(t, err)
	require.NotContains(t, string(raw), "anthropic_version",
		"DeepSeek body 绝不能包含 anthropic_version")
	require.NotContains(t, string(raw), "\"thinking\"",
		"DeepSeek 不使用 GLM 的 thinking 字段")
}

func TestConvertToDeepSeekRequest_ContentRawMessagePassthrough(t *testing.T) {
	t.Parallel()
	req := &dto.GeneralOpenAIRequest{
		Model: "deepseek.v3.2",
		Messages: []dto.Message{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
	}
	ds := convertToDeepSeekRequest(req)
	require.Len(t, ds.Messages, 1)
	require.Equal(t, `"hi"`, string(ds.Messages[0].Content))
}

func TestConvertToDeepSeekRequest_NilContentDropsField(t *testing.T) {
	t.Parallel()
	req := &dto.GeneralOpenAIRequest{
		Model: "deepseek.v3.2",
		Messages: []dto.Message{
			{Role: "assistant", Content: nil},
		},
	}
	ds := convertToDeepSeekRequest(req)
	require.Len(t, ds.Messages, 1)
	require.Empty(t, ds.Messages[0].Content,
		"nil content 必须让 DeepSeekMessage.Content 留空，omitempty 才能丢弃")

	raw, err := common.Marshal(ds)
	require.NoError(t, err)
	require.NotContains(t, string(raw), `"content":null`,
		"绝不能输出字面 null content")
}
