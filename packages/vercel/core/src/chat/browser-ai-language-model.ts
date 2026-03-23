import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  SharedV3Warning,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3ProviderTool,
  LanguageModelV3StreamPart,
  LanguageModelV3ToolCall,
  JSONValue,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import {
  buildJsonToolSystemPrompt,
  parseJsonFunctionCalls,
  createUnsupportedToolWarning,
  isFunctionTool,
  processToolCallStream,
  type DownloadProgressCallback,
} from "@browser-ai/shared";
import { convertToBrowserAIMessages } from "../utils/convert-to-browser-ai-messages";
import { gatherUnsupportedSettingWarnings } from "../utils/warnings";
import { getMultimodalInfo } from "../utils/prompt-utils";
import { SessionManager } from "./session-manager";

export type BrowserAIChatModelId = "text";

export interface BrowserAIChatSettings extends Omit<
  LanguageModelCreateOptions,
  "initialPrompts"
> {
  /**
   * Expected input types for the session, for multimodal inputs.
   */
  expectedInputs?: Array<{
    type: "text" | "image" | "audio";
    languages?: string[];
  }>;

  /**
   * Callback invoked when the model context window is exceeded.
   * Replaces the deprecated `onQuotaOverflow`.
   * @see [Prompt API Context Overflow](https://github.com/webmachinelearning/prompt-api?tab=readme-ov-file#tokenization-context-window-length-limits-and-overflow)
   */
  onContextOverflow?: (event: Event) => void;

  /**
   * @deprecated Use `onContextOverflow` instead.
   * @see [Prompt API Quota Overflow](https://github.com/webmachinelearning/prompt-api?tab=readme-ov-file#tokenization-context-window-length-limits-and-overflow)
   * @param event
   */
  onQuotaOverflow?: (event: Event) => void;
}

/**
 * Check if the browser supports the browser AI API
 * @returns true if the browser supports the browser AI API, false otherwise
 */
export function doesBrowserSupportBrowserAI(): boolean {
  return typeof LanguageModel !== "undefined";
}

type BrowserAIConfig = {
  provider: string;
  modelId: BrowserAIChatModelId;
  options: BrowserAIChatSettings;
};

export class BrowserAIChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly modelId: BrowserAIChatModelId;
  readonly provider = "browser-ai";

  private readonly config: BrowserAIConfig;
  private readonly sessionManager: SessionManager;

  constructor(
    modelId: BrowserAIChatModelId,
    options: BrowserAIChatSettings = {},
  ) {
    this.modelId = modelId;
    this.config = {
      provider: this.provider,
      modelId,
      options,
    };
    this.sessionManager = new SessionManager(options);
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^https?:\/\/.+$/],
    "audio/*": [/^https?:\/\/.+$/],
  };

  /**
   * Gets a session with the specified options
   * Delegates to SessionManager for all session lifecycle management
   * @private
   */
  private async getSession(
    options?: LanguageModelCreateOptions,
    expectedInputs?: Array<{ type: "text" | "image" | "audio" }>,
    systemMessage?: string,
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<LanguageModel> {
    return this.sessionManager.getSession({
      ...options,
      expectedInputs,
      systemMessage,
      onDownloadProgress,
    });
  }

  private getArgs(callOptions: Parameters<LanguageModelV3["doGenerate"]>[0]) {
    const {
      prompt,
      maxOutputTokens,
      temperature,
      topP,
      topK,
      frequencyPenalty,
      presencePenalty,
      stopSequences,
      responseFormat,
      seed,
      tools,
      toolChoice,
      providerOptions,
    } = callOptions;
    const warnings: SharedV3Warning[] = [];

    // Gather warnings for unsupported settings
    warnings.push(
      ...gatherUnsupportedSettingWarnings({
        maxOutputTokens,
        stopSequences,
        topP,
        presencePenalty,
        frequencyPenalty,
        seed,
        toolChoice,
      }),
    );

    // Filter and warn about unsupported tools
    const functionTools = (tools ?? []).filter(isFunctionTool);

    const unsupportedTools = (tools ?? []).filter(
      (tool): tool is LanguageModelV3ProviderTool => !isFunctionTool(tool),
    );

    for (const tool of unsupportedTools) {
      warnings.push(
        createUnsupportedToolWarning(
          tool,
          "Only function tools are supported by the Prompt API polyfill",
        ),
      );
    }

    // Detect multimodal content and collect expected inputs in one pass
    const { hasMultiModalInput, expectedInputs } = getMultimodalInfo(prompt);

    // Convert messages to the DOM API format
    const { systemMessage, messages } = convertToBrowserAIMessages(prompt);

    // Handle response format for Prompt API
    const promptOptions: LanguageModelPromptOptions &
      LanguageModelCreateCoreOptions = {};
    if (responseFormat?.type === "json") {
      promptOptions.responseConstraint = responseFormat.schema as Record<
        string,
        JSONValue
      >;
    }

    // Map supported settings
    if (temperature !== undefined) {
      promptOptions.temperature = temperature;
    }

    if (topK !== undefined) {
      promptOptions.topK = topK;
    }

    return {
      systemMessage,
      messages,
      warnings,
      promptOptions,
      hasMultiModalInput,
      expectedInputs,
      functionTools,
    };
  }

  /**
   * Generates a complete text response using the browser's built-in Prompt API
   * @param options
   * @returns Promise resolving to the generated content with finish reason, usage stats, and any warnings
   * @throws {LoadSettingError} When the Prompt API is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const converted = this.getArgs(options);
    const {
      systemMessage,
      messages,
      warnings,
      promptOptions,
      expectedInputs,
      functionTools,
    } = converted;

    const systemPrompt = buildJsonToolSystemPrompt(
      systemMessage,
      functionTools,
      {
        allowParallelToolCalls: false,
      },
    );

    const session = await this.getSession(
      { signal: options.abortSignal },
      expectedInputs,
      systemPrompt || undefined,
    );

    const promptCallOptions = {
      ...promptOptions,
      signal: options.abortSignal,
    };

    const rawResponse = await session.prompt(messages, promptCallOptions);

    // Parse JSON tool calls from response
    const { toolCalls, textContent } = parseJsonFunctionCalls(rawResponse);

    if (toolCalls.length > 0) {
      const toolCallsToEmit = toolCalls.slice(0, 1);

      const parts: LanguageModelV3Content[] = [];

      if (textContent) {
        parts.push({
          type: "text",
          text: textContent,
        });
      }

      for (const call of toolCallsToEmit) {
        parts.push({
          type: "tool-call",
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: JSON.stringify(call.args ?? {}),
        } satisfies LanguageModelV3ToolCall);
      }

      return {
        content: parts,
        finishReason: { unified: "tool-calls", raw: "tool-calls" },
        usage: {
          inputTokens: {
            total: undefined,
            noCache: undefined,
            cacheRead: undefined,
            cacheWrite: undefined,
          },
          outputTokens: {
            total: undefined,
            text: undefined,
            reasoning: undefined,
          },
        },
        request: { body: { messages, options: promptOptions } },
        warnings,
      };
    }

    const content: LanguageModelV3Content[] = [
      {
        type: "text",
        text: textContent || rawResponse,
      },
    ];

    return {
      content,
      finishReason: { unified: "stop", raw: "stop" },
      usage: {
        inputTokens: {
          total: undefined,
          noCache: undefined,
          cacheRead: undefined,
          cacheWrite: undefined,
        },
        outputTokens: {
          total: undefined,
          text: undefined,
          reasoning: undefined,
        },
      },
      request: { body: { messages, options: promptOptions } },
      warnings,
    };
  }

  /**
   * Gets the current context usage (tokens consumed) for the current session, if available
   * @returns The context usage or undefined if not available
   */
  public getContextUsage(): number | undefined {
    return this.sessionManager.getContextUsage();
  }

  /**
   * Gets the context window size (token limit) for the current session, if available
   * @returns The context window size or undefined if not available
   */
  public getContextWindow(): number | undefined {
    return this.sessionManager.getContextWindow();
  }

  /**
   * @deprecated Use {@link getContextUsage} instead.
   */
  public getInputUsage(): number | undefined {
    return this.getContextUsage();
  }

  /**
   * @deprecated Use {@link getContextWindow} instead.
   */
  public getInputQuota(): number | undefined {
    return this.getContextWindow();
  }

  /**
   * Check the availability of the browser AI model
   * @returns Promise resolving to "unavailable", "available", or "available-after-download"
   */
  public async availability(): Promise<Availability> {
    return this.sessionManager.checkAvailability();
  }

  /**
   * Creates a session with download progress monitoring.
   *
   * @example
   * ```typescript
   * const session = await model.createSessionWithProgress(
   *   (progress) => {
   *     console.log(`Download progress: ${Math.round(progress * 100)}%`);
   *   }
   * );
   * ```
   *
   * @param onDownloadProgress Optional callback receiving progress values from 0 to 1
   * @returns Promise resolving to the model instance
   * @throws {LoadSettingError} When the Prompt API is not available or model is unavailable
   */
  public async createSessionWithProgress(
    onDownloadProgress?: DownloadProgressCallback,
  ): Promise<BrowserAIChatLanguageModel> {
    await this.sessionManager.createSessionWithProgress(onDownloadProgress);
    return this;
  }

  /**
   * Generates a streaming text response using the browser's built-in Prompt API
   * @param options
   * @returns Promise resolving to a readable stream of text chunks and request metadata
   * @throws {LoadSettingError} When the Prompt API is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const converted = this.getArgs(options);
    const {
      systemMessage,
      messages,
      warnings,
      promptOptions,
      expectedInputs,
      functionTools,
    } = converted;

    // Build system prompt with JSON tool calling instructions
    const systemPrompt = buildJsonToolSystemPrompt(
      systemMessage,
      functionTools,
      {
        allowParallelToolCalls: false,
      },
    );

    const session = await this.getSession(
      { signal: options.abortSignal },
      expectedInputs,
      systemPrompt || undefined,
    );

    // Pass abort signal to the native streaming method
    const streamOptions = {
      ...promptOptions,
      signal: options.abortSignal,
    };
    const conversationHistory = [...messages];
    const textId = "text-0";

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: async (controller) => {
        controller.enqueue({
          type: "stream-start",
          warnings,
        });

        let textStarted = false;
        let finished = false;
        let aborted = false;
        let currentReader: ReadableStreamDefaultReader<string> | null = null;

        const ensureTextStart = () => {
          if (!textStarted) {
            controller.enqueue({
              type: "text-start",
              id: textId,
            });
            textStarted = true;
          }
        };

        const emitTextDelta = (delta: string) => {
          if (!delta) return;
          ensureTextStart();
          controller.enqueue({
            type: "text-delta",
            id: textId,
            delta,
          });
        };

        const emitTextEndIfNeeded = () => {
          if (!textStarted) return;
          controller.enqueue({
            type: "text-end",
            id: textId,
          });
          textStarted = false;
        };

        const finishStream = (finishReason: LanguageModelV3FinishReason) => {
          if (finished) return;
          finished = true;
          emitTextEndIfNeeded();
          controller.enqueue({
            type: "finish",
            finishReason,
            usage: {
              inputTokens: {
                total:
                  (session as LanguageModel).contextUsage ?? session.inputUsage,
                noCache: undefined,
                cacheRead: undefined,
                cacheWrite: undefined,
              },
              outputTokens: {
                total: undefined,
                text: undefined,
                reasoning: undefined,
              },
            },
          });
          controller.close();
        };

        const abortHandler = () => {
          if (aborted) {
            return;
          }
          aborted = true;
          if (currentReader) {
            currentReader.cancel().catch(() => undefined);
          }
          finishStream({ unified: "stop", raw: "aborted" });
        };

        if (options.abortSignal) {
          options.abortSignal.addEventListener("abort", abortHandler);
        }

        try {
          const promptStream = session.promptStreaming(
            conversationHistory,
            streamOptions,
          );
          currentReader = promptStream.getReader();

          const chunks = (async function* () {
            while (!aborted) {
              const { done, value } = await currentReader!.read();
              if (done) break;
              yield value;
            }
          })();

          const result = await processToolCallStream(
            chunks,
            emitTextDelta,
            controller,
            { stopEarlyOnToolCall: true },
          );

          if (result.toolCallDetected && currentReader) {
            await currentReader.cancel().catch(() => undefined);
          }
          currentReader = null;

          if (aborted) {
            return;
          }

          if (!result.toolCallDetected || result.toolCalls.length === 0) {
            finishStream({ unified: "stop", raw: "stop" });
            return;
          }

          if (result.trailingText) {
            emitTextDelta(result.trailingText);
          }

          finishStream({ unified: "tool-calls", raw: "tool-calls" });
        } catch (error) {
          controller.enqueue({ type: "error", error });
          controller.close();
        } finally {
          if (options.abortSignal) {
            options.abortSignal.removeEventListener("abort", abortHandler);
          }
        }
      },
    });

    return {
      stream,
      request: { body: { messages, options: promptOptions } },
    };
  }
}
