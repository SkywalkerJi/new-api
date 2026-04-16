package aws

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/relay/channel"
	"github.com/QuantumNous/new-api/relay/channel/claude"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"

	"github.com/QuantumNous/new-api/setting/model_setting"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrockruntimeTypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/auth/bearer"
)

// getAwsErrorStatusCode extracts HTTP status code from AWS SDK error
func getAwsErrorStatusCode(err error) int {
	// Check for HTTP response error which contains status code
	var httpErr interface{ HTTPStatusCode() int }
	if errors.As(err, &httpErr) {
		return httpErr.HTTPStatusCode()
	}
	// Default to 500 if we can't determine the status code
	return http.StatusInternalServerError
}

func newAwsInvokeContext() (context.Context, context.CancelFunc) {
	if common.RelayTimeout <= 0 {
		return context.Background(), func() {}
	}
	return context.WithTimeout(context.Background(), time.Duration(common.RelayTimeout)*time.Second)
}

func newAwsClient(c *gin.Context, info *relaycommon.RelayInfo) (*bedrockruntime.Client, error) {
	var (
		httpClient *http.Client
		err        error
	)
	if info.ChannelSetting.Proxy != "" {
		httpClient, err = service.NewProxyHttpClient(info.ChannelSetting.Proxy)
		if err != nil {
			return nil, fmt.Errorf("new proxy http client failed: %w", err)
		}
	} else {
		httpClient = service.GetHttpClient()
	}

	awsSecret := strings.Split(info.ApiKey, "|")
	var client *bedrockruntime.Client
	switch len(awsSecret) {
	case 2:
		apiKey := awsSecret[0]
		region := awsSecret[1]
		client = bedrockruntime.New(bedrockruntime.Options{
			Region:                  region,
			BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
			HTTPClient:              httpClient,
		})
	case 3:
		ak := awsSecret[0]
		sk := awsSecret[1]
		region := awsSecret[2]
		client = bedrockruntime.New(bedrockruntime.Options{
			Region:      region,
			Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(ak, sk, "")),
			HTTPClient:  httpClient,
		})
	default:
		return nil, errors.New("invalid aws secret key")
	}

	return client, nil
}

func doAwsClientRequest(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor, requestBody io.Reader) (any, error) {
	awsCli, err := newAwsClient(c, info)
	if err != nil {
		return nil, types.NewError(err, types.ErrorCodeChannelAwsClientError)
	}
	a.AwsClient = awsCli

	// 获取对应的AWS模型ID
	awsModelId := resolveAwsModelId(info.UpstreamModelName, awsCli.Options().Region)

	// init empty request.header
	requestHeader := http.Header{}
	a.SetupRequestHeader(c, &requestHeader, info)
	headerOverride, err := channel.ResolveHeaderOverride(info, c)
	if err != nil {
		return nil, err
	}
	for key, value := range headerOverride {
		requestHeader.Set(key, value)
	}

	if isNovaModel(awsModelId) {
		var novaReq *NovaRequest
		err = common.DecodeJson(requestBody, &novaReq)
		if err != nil {
			return nil, types.NewError(errors.Wrap(err, "decode nova request fail"), types.ErrorCodeBadRequestBody)
		}

		// 使用InvokeModel API，但使用Nova格式的请求体
		awsReq := &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(awsModelId),
			Accept:      aws.String("application/json"),
			ContentType: aws.String("application/json"),
		}

		reqBody, err := common.Marshal(novaReq)
		if err != nil {
			return nil, types.NewError(errors.Wrap(err, "marshal nova request"), types.ErrorCodeBadResponseBody)
		}
		awsReq.Body = reqBody
		a.AwsReq = awsReq
		return nil, nil
	}

	// NOTE: `isGlmModel(awsModelId) || a.IsGlm` is defensive — the flag is
	// set by ConvertOpenAIRequest for OpenAI-format entry, and the
	// model-id predicate covers Claude-native entry paths and test call sites
	// that construct Adaptor directly.
	if isGlmModel(awsModelId) || a.IsGlm {
		bodyBytes, err := io.ReadAll(requestBody)
		if err != nil {
			return nil, types.NewError(errors.Wrap(err, "read glm request body"), types.ErrorCodeBadRequestBody)
		}
		if info.IsStream {
			a.AwsReq = &bedrockruntime.InvokeModelWithResponseStreamInput{
				ModelId:     aws.String(awsModelId),
				Accept:      aws.String("application/json"),
				ContentType: aws.String("application/json"),
				Body:        bodyBytes,
			}
		} else {
			a.AwsReq = &bedrockruntime.InvokeModelInput{
				ModelId:     aws.String(awsModelId),
				Accept:      aws.String("application/json"),
				ContentType: aws.String("application/json"),
				Body:        bodyBytes,
			}
		}
		return nil, nil
	}

	// NOTE: `isDeepSeekModel(awsModelId) || a.IsDeepSeek` is defensive — same
	// pattern as the GLM branch above. The flag is set by ConvertOpenAIRequest
	// for OpenAI-format entry; the model-id predicate covers Claude-native
	// entry paths and test callers that construct Adaptor directly.
	// DeepSeek V3.2 uses the same OpenAI Chat Completions schema as GLM, but
	// is a physically independent branch so the two families can diverge.
	if isDeepSeekModel(awsModelId) || a.IsDeepSeek {
		bodyBytes, err := io.ReadAll(requestBody)
		if err != nil {
			return nil, types.NewError(errors.Wrap(err, "read deepseek request body"), types.ErrorCodeBadRequestBody)
		}
		if info.IsStream {
			a.AwsReq = &bedrockruntime.InvokeModelWithResponseStreamInput{
				ModelId:     aws.String(awsModelId),
				Accept:      aws.String("application/json"),
				ContentType: aws.String("application/json"),
				Body:        bodyBytes,
			}
		} else {
			a.AwsReq = &bedrockruntime.InvokeModelInput{
				ModelId:     aws.String(awsModelId),
				Accept:      aws.String("application/json"),
				ContentType: aws.String("application/json"),
				Body:        bodyBytes,
			}
		}
		return nil, nil
	}

	body, err := buildClaudeNativeBody(c, info, requestBody, requestHeader)
	if err != nil {
		return nil, types.NewError(err, types.ErrorCodeBadRequestBody)
	}
	if info.IsStream {
		a.AwsReq = &bedrockruntime.InvokeModelWithResponseStreamInput{
			ModelId:     aws.String(awsModelId),
			Accept:      aws.String("application/json"),
			ContentType: aws.String("application/json"),
			Body:        body,
		}
	} else {
		a.AwsReq = &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(awsModelId),
			Accept:      aws.String("application/json"),
			ContentType: aws.String("application/json"),
			Body:        body,
		}
	}
	return nil, nil
}

// buildClaudeNativeBody 复用 formatRequest + buildAwsRequestBody 产出 Bedrock 可接收的
// Claude native JSON，带 anthropic_version 和 anthropic-beta header 处理。
// 两种认证模式（API Key / AKSK）都应调用它产出最终 body。
func buildClaudeNativeBody(c *gin.Context, info *relaycommon.RelayInfo, requestBody io.Reader, requestHeader http.Header) ([]byte, error) {
	awsClaudeReq, err := formatRequest(requestBody, requestHeader)
	if err != nil {
		return nil, errors.Wrap(err, "format aws request fail")
	}
	return buildAwsRequestBody(c, info, awsClaudeReq)
}

// buildAwsRequestBody prepares the payload for AWS requests, applying passthrough rules when enabled.
func buildAwsRequestBody(c *gin.Context, info *relaycommon.RelayInfo, awsClaudeReq any) ([]byte, error) {
	if model_setting.GetGlobalSettings().PassThroughRequestEnabled || info.ChannelSetting.PassThroughBodyEnabled {
		storage, err := common.GetBodyStorage(c)
		if err != nil {
			return nil, errors.Wrap(err, "get request body for pass-through fail")
		}
		body, err := storage.Bytes()
		if err != nil {
			return nil, errors.Wrap(err, "get request body bytes fail")
		}
		var data map[string]interface{}
		if err := common.Unmarshal(body, &data); err != nil {
			return nil, errors.Wrap(err, "pass-through unmarshal request body fail")
		}
		delete(data, "model")
		delete(data, "stream")
		return common.Marshal(data)
	}
	return common.Marshal(awsClaudeReq)
}

func getAwsRegionPrefix(awsRegionId string) string {
	parts := strings.Split(awsRegionId, "-")
	regionPrefix := ""
	if len(parts) > 0 {
		regionPrefix = parts[0]
	}
	return regionPrefix
}

func awsModelCanCrossRegion(awsModelId, awsRegionPrefix string) bool {
	regionSet, exists := awsModelCanCrossRegionMap[awsModelId]
	return exists && regionSet[awsRegionPrefix]
}

func awsModelCrossRegion(awsModelId, awsRegionPrefix, awsRegion string) string {
	if p, ok := awsRegionFineGrainedPrefixMap[awsRegion]; ok {
		return p + "." + awsModelId
	}
	modelPrefix, find := awsRegionCrossModelPrefixMap[awsRegionPrefix]
	if !find {
		return awsModelId
	}
	return modelPrefix + "." + awsModelId
}

func getAwsModelID(requestModel string) string {
	if awsModelIDName, ok := awsModelIDMap[requestModel]; ok {
		return awsModelIDName
	}
	return requestModel
}

// resolveAwsModelId 把前端传入的请求模型名解析成最终发给 Bedrock 的 model id。
// 统一应用 alias 映射、global 强制前缀、国家级前缀和大洲级前缀。
// 两种认证模式（API Key / AKSK）都应通过它产出 modelId。
func resolveAwsModelId(requestModel, awsRegion string) string {
	awsModelId := getAwsModelID(requestModel)
	if awsModelForceGlobalMap[awsModelId] {
		return "global." + awsModelId
	}
	awsRegionPrefix := getAwsRegionPrefix(awsRegion)
	if !awsModelCanCrossRegion(awsModelId, awsRegionPrefix) {
		return awsModelId
	}
	return awsModelCrossRegion(awsModelId, awsRegionPrefix, awsRegion)
}

func awsHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {

	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModel(ctx, a.AwsReq.(*bedrockruntime.InvokeModelInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModel"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}

	claudeInfo := &claude.ClaudeResponseInfo{
		ResponseId:   helper.GetResponseID(c),
		Created:      common.GetTimestamp(),
		Model:        info.UpstreamModelName,
		ResponseText: strings.Builder{},
		Usage:        &dto.Usage{},
	}

	// 复制上游 Content-Type 到客户端响应头
	if awsResp.ContentType != nil && *awsResp.ContentType != "" {
		c.Writer.Header().Set("Content-Type", *awsResp.ContentType)
	}

	handlerErr := claude.HandleClaudeResponseData(c, info, claudeInfo, nil, awsResp.Body)
	if handlerErr != nil {
		return handlerErr, nil
	}
	return nil, claudeInfo.Usage
}

func awsStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {
	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModelWithResponseStream(ctx, a.AwsReq.(*bedrockruntime.InvokeModelWithResponseStreamInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModelWithResponseStream"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}
	stream := awsResp.GetStream()
	defer stream.Close()

	claudeInfo := &claude.ClaudeResponseInfo{
		ResponseId:   helper.GetResponseID(c),
		Created:      common.GetTimestamp(),
		Model:        info.UpstreamModelName,
		ResponseText: strings.Builder{},
		Usage:        &dto.Usage{},
	}

	for event := range stream.Events() {
		switch v := event.(type) {
		case *bedrockruntimeTypes.ResponseStreamMemberChunk:
			info.SetFirstResponseTime()
			respErr := claude.HandleStreamResponseData(c, info, claudeInfo, string(v.Value.Bytes))
			if respErr != nil {
				return respErr, nil
			}
		case *bedrockruntimeTypes.UnknownUnionMember:
			common.SysError(fmt.Sprintf("aws bedrock unknown stream tag: %s", v.Tag))
			return types.NewError(errors.New("unknown response type"), types.ErrorCodeInvalidRequest), nil
		default:
			common.SysError("aws bedrock received nil or unknown stream event type")
			return types.NewError(errors.New("nil or unknown response type"), types.ErrorCodeInvalidRequest), nil
		}
	}

	claude.HandleStreamFinalResponse(c, info, claudeInfo)
	return nil, claudeInfo.Usage
}

// Nova模型处理函数
func handleNovaRequest(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {

	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModel(ctx, a.AwsReq.(*bedrockruntime.InvokeModelInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModel"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}

	// 解析Nova响应
	var novaResp struct {
		Output struct {
			Message struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"message"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"inputTokens"`
			OutputTokens int `json:"outputTokens"`
			TotalTokens  int `json:"totalTokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(awsResp.Body, &novaResp); err != nil {
		return types.NewError(errors.Wrap(err, "unmarshal nova response"), types.ErrorCodeBadResponseBody), nil
	}

	// 构造OpenAI格式响应
	response := dto.OpenAITextResponse{
		Id:      helper.GetResponseID(c),
		Object:  "chat.completion",
		Created: common.GetTimestamp(),
		Model:   info.UpstreamModelName,
		Choices: []dto.OpenAITextResponseChoice{{
			Index: 0,
			Message: dto.Message{
				Role:    "assistant",
				Content: novaResp.Output.Message.Content[0].Text,
			},
			FinishReason: "stop",
		}},
		Usage: dto.Usage{
			PromptTokens:     novaResp.Usage.InputTokens,
			CompletionTokens: novaResp.Usage.OutputTokens,
			TotalTokens:      novaResp.Usage.TotalTokens,
		},
	}

	c.JSON(http.StatusOK, response)
	return nil, &response.Usage
}

// decodeGlmNonStreamBody parses a GLM non-stream response body (OpenAI chat.completion JSON),
// writes it to the gin context and returns extracted usage.
// Extracted as a standalone helper to enable unit testing without mocking the SDK.
func decodeGlmNonStreamBody(c *gin.Context, info *relaycommon.RelayInfo, body []byte) (*dto.Usage, *types.NewAPIError) {
	// Only pull out usage for billing; forward the rest of the body verbatim so that
	// provider-specific fields (thinking / reasoning_content / tool_calls / …) are preserved.
	var parsed struct {
		Usage dto.Usage `json:"usage"`
	}
	if err := common.Unmarshal(body, &parsed); err != nil {
		return nil, types.NewError(errors.Wrap(err, "unmarshal glm response"), types.ErrorCodeBadResponseBody)
	}
	c.Data(http.StatusOK, "application/json", body)
	return &parsed.Usage, nil
}

func awsGlmHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {
	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModel(ctx, a.AwsReq.(*bedrockruntime.InvokeModelInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModel"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}
	usage, apiErr := decodeGlmNonStreamBody(c, info, awsResp.Body)
	return apiErr, usage
}

func awsGlmStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {
	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModelWithResponseStream(ctx, a.AwsReq.(*bedrockruntime.InvokeModelWithResponseStreamInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModelWithResponseStream"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}
	stream := awsResp.GetStream()
	defer stream.Close()

	helper.SetEventStreamHeaders(c)

	var usage dto.Usage
	for event := range stream.Events() {
		switch v := event.(type) {
		case *bedrockruntimeTypes.ResponseStreamMemberChunk:
			info.SetFirstResponseTime()
			chunkBytes := v.Value.Bytes
			if len(chunkBytes) == 0 {
				continue // skip empty keep-alive / heartbeat
			}
			// Extract usage if present (usually on the final chunk)
			var partial struct {
				Usage *dto.Usage `json:"usage"`
			}
			_ = common.Unmarshal(chunkBytes, &partial)
			if partial.Usage != nil {
				usage = *partial.Usage
			}
			// Forward as SSE via project helper (framing + flush in one shot).
			if werr := helper.StringData(c, string(chunkBytes)); werr != nil {
				// Write failures indicate the client disconnected; log and exit quietly
				// with accumulated usage rather than propagating as an HTTP error.
				// Billing semantics: the partial usage accumulated up to the disconnect
				// point is returned as-is and will be charged. Upstream token cost is
				// incurred regardless of whether the client received the full stream.
				common.SysError("aws glm stream write failed: " + werr.Error())
				return nil, &usage
			}
		case *bedrockruntimeTypes.UnknownUnionMember:
			common.SysError(fmt.Sprintf("aws bedrock unknown glm stream tag: %s", v.Tag))
			return types.NewError(errors.New("unknown response type"), types.ErrorCodeInvalidRequest), nil
		default:
			common.SysError("aws bedrock received nil or unknown glm stream event type")
			return types.NewError(errors.New("nil or unknown response type"), types.ErrorCodeInvalidRequest), nil
		}
	}
	helper.Done(c)
	return nil, &usage
}

// decodeDeepSeekNonStreamBody 解析 DeepSeek 非流式响应（OpenAI chat.completion JSON），
// 写回 gin 上下文并提取 usage。作为独立 helper 便于无 SDK mock 的单测。
func decodeDeepSeekNonStreamBody(c *gin.Context, info *relaycommon.RelayInfo, body []byte) (*dto.Usage, *types.NewAPIError) {
	var parsed struct {
		Usage dto.Usage `json:"usage"`
	}
	if err := common.Unmarshal(body, &parsed); err != nil {
		return nil, types.NewError(errors.Wrap(err, "unmarshal deepseek response"), types.ErrorCodeBadResponseBody)
	}
	c.Data(http.StatusOK, "application/json", body)
	return &parsed.Usage, nil
}

func awsDeepSeekHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {
	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModel(ctx, a.AwsReq.(*bedrockruntime.InvokeModelInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModel"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}
	usage, apiErr := decodeDeepSeekNonStreamBody(c, info, awsResp.Body)
	return apiErr, usage
}

func awsDeepSeekStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, a *Adaptor) (*types.NewAPIError, *dto.Usage) {
	ctx, cancel := newAwsInvokeContext()
	defer cancel()

	awsResp, err := a.AwsClient.InvokeModelWithResponseStream(ctx, a.AwsReq.(*bedrockruntime.InvokeModelWithResponseStreamInput))
	if err != nil {
		statusCode := getAwsErrorStatusCode(err)
		return types.NewOpenAIError(errors.Wrap(err, "InvokeModelWithResponseStream"), types.ErrorCodeAwsInvokeError, statusCode), nil
	}
	stream := awsResp.GetStream()
	defer stream.Close()

	helper.SetEventStreamHeaders(c)

	var usage dto.Usage
	for event := range stream.Events() {
		switch v := event.(type) {
		case *bedrockruntimeTypes.ResponseStreamMemberChunk:
			info.SetFirstResponseTime()
			chunkBytes := v.Value.Bytes
			if len(chunkBytes) == 0 {
				continue
			}
			var partial struct {
				Usage *dto.Usage `json:"usage"`
			}
			_ = common.Unmarshal(chunkBytes, &partial)
			if partial.Usage != nil {
				usage = *partial.Usage
			}
			if werr := helper.StringData(c, string(chunkBytes)); werr != nil {
				// Billing 语义：断连时已累积的 usage 原样返回并计费，
				// 上游 token 成本无论客户端是否收完流都已产生。
				common.SysError("aws deepseek stream write failed: " + werr.Error())
				return nil, &usage
			}
		case *bedrockruntimeTypes.UnknownUnionMember:
			common.SysError(fmt.Sprintf("aws bedrock unknown deepseek stream tag: %s", v.Tag))
			return types.NewError(errors.New("unknown response type"), types.ErrorCodeInvalidRequest), nil
		default:
			common.SysError("aws bedrock received nil or unknown deepseek stream event type")
			return types.NewError(errors.New("nil or unknown response type"), types.ErrorCodeInvalidRequest), nil
		}
	}
	helper.Done(c)
	return nil, &usage
}
