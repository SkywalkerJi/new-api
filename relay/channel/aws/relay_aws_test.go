package aws

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
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
