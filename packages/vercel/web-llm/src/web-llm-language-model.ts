import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  SharedV3Warning,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3ProviderTool,
  LanguageModelV3StreamPart,
  LanguageModelV3ToolCall,
  LoadSettingError,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import { convertToWebLLMMessages } from "./convert-to-webllm-messages";

import {
  AppConfig,
  ChatCompletionRequestStreaming,
  CreateWebWorkerMLCEngine,
  InitProgressReport,
  MLCEngine,
  MLCEngineConfig,
  MLCEngineInterface,
} from "@mlc-ai/web-llm";
import { Availability } from "./types";
import {
  buildJsonToolSystemPrompt,
  parseJsonFunctionCalls,
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  isFunctionTool,
  ToolCallFenceDetector,
  type ToolDefinition,
} from "@browser-ai/shared";
import {
  prependSystemPromptToMessages,
  extractSystemPrompt,
} from "./utils/prompt-utils";

declare global {
  interface Navigator {
    gpu?: GPU;
  }
}

export type WebLLMModelId = string;

export interface WebLLMSettings {
  /**
   * Custom app configuration for WebLLM
   */
  appConfig?: AppConfig;
  /**
   * Progress callback for model initialization
   */
  initProgressCallback?: (progress: InitProgressReport) => void;
  /**
   * Engine configuration options
   */
  engineConfig?: MLCEngineConfig;
  /**
   * A web worker instance to run the model in.
   * When provided, the model will run in a separate thread.
   *
   * @default undefined
   */
  worker?: Worker;
}

function isMobile(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  );
}

function checkWebGPU(): boolean {
  try {
    return !!globalThis?.navigator?.gpu;
  } catch {
    return false;
  }
}

/**
 * Check if the browser supports WebGPU (required for WebLLM).
 * @returns boolean - true if WebGPU API is available
 */
export function doesBrowserSupportWebLLM(): boolean {
  return checkWebGPU();
}

function extractToolName(content: string): string | null {
  // For JSON mode: {"name":"toolName"
  const jsonMatch = content.match(/\{\s*"name"\s*:\s*"([^"]+)"/);
  if (jsonMatch) {
    return jsonMatch[1];
  }
  return null;
}

function extractArgumentsContent(content: string): string {
  const match = content.match(/"arguments"\s*:\s*/);
  if (!match || match.index === undefined) {
    return "";
  }

  const startIndex = match.index + match[0].length;
  let result = "";
  let depth = 0;
  let inString = false;
  let escaped = false;
  let started = false;

  for (let i = startIndex; i < content.length; i++) {
    const char = content[i];
    result += char;

    if (!started) {
      if (!/\s/.test(char)) {
        started = true;
        if (char === "{" || char === "[") {
          depth = 1;
        }
      }
      continue;
    }

    if (escaped) {
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === '"') {
      inString = !inString;
      continue;
    }

    if (!inString) {
      if (char === "{" || char === "[") {
        depth += 1;
      } else if (char === "}" || char === "]") {
        if (depth > 0) {
          depth -= 1;
          if (depth === 0) {
            break;
          }
        }
      }
    }
  }

  return result;
}

type WebLLMConfig = {
  provider: string;
  modelId: WebLLMModelId;
  options: WebLLMSettings;
};

export class WebLLMLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly modelId: WebLLMModelId;
  readonly provider = "web-llm";

  private readonly config: WebLLMConfig;
  private engine?: MLCEngineInterface;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;

  constructor(modelId: WebLLMModelId, options: WebLLMSettings = {}) {
    this.modelId = modelId;
    this.config = {
      provider: this.provider,
      modelId,
      options,
    };
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    // WebLLM doesn't support URLs natively
  };

  /**
   * Check if the model is initialized and ready to use
   * @returns true if the model is initialized, false otherwise
   */
  get isModelInitialized(): boolean {
    return this.isInitialized;
  }

  private async getEngine(
    options?: MLCEngineConfig,
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<MLCEngineInterface> {
    const availability = await this.availability();
    if (availability === "unavailable") {
      throw new LoadSettingError({
        message:
          "WebLLM is not available. This library requires a browser with WebGPU support.",
      });
    }

    if (this.engine && this.isInitialized) return this.engine;

    // If initialization is already in progress, wait for it
    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.engine) return this.engine;
    }

    this.initializationPromise = this._initializeEngine(
      options,
      onInitProgress,
    );
    await this.initializationPromise;

    if (!this.engine) {
      throw new LoadSettingError({
        message: "Engine initialization failed",
      });
    }

    return this.engine;
  }

  private async _initializeEngine(
    options?: MLCEngineConfig,
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<void> {
    try {
      // Create engine instance
      const engineConfig = {
        ...this.config.options.engineConfig,
        ...options,
        initProgressCallback:
          onInitProgress || this.config.options.initProgressCallback,
      };

      if (this.config.options.worker) {
        this.engine = await CreateWebWorkerMLCEngine(
          this.config.options.worker,
          this.modelId,
          engineConfig,
        );
      } else {
        this.engine = new MLCEngine(engineConfig);
        // Load the model
        await this.engine.reload(this.modelId);
      }

      this.isInitialized = true;
    } catch (error) {
      // Reset state on error so we can retry
      this.engine = undefined;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize WebLLM engine: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }

  private getArgs({
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
  }: Parameters<LanguageModelV3["doGenerate"]>[0]) {
    const warnings: SharedV3Warning[] = [];

    const functionTools: ToolDefinition[] = (tools ?? [])
      .filter(isFunctionTool)
      .map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      }));

    const unsupportedTools = (tools ?? []).filter(
      (tool): tool is LanguageModelV3ProviderTool => !isFunctionTool(tool),
    );

    for (const tool of unsupportedTools) {
      warnings.push(
        createUnsupportedToolWarning(
          tool,
          "Only function tools are supported by WebLLM",
        ),
      );
    }

    if (topK != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "topK",
          "topK is not supported by WebLLM",
        ),
      );
    }

    if (stopSequences != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "stopSequences",
          "Stop sequences may not be fully implemented",
        ),
      );
    }

    if (presencePenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "presencePenalty",
          "Presence penalty is not fully implemented",
        ),
      );
    }

    if (frequencyPenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "frequencyPenalty",
          "Frequency penalty is not fully implemented",
        ),
      );
    }

    if (toolChoice != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "toolChoice",
          "toolChoice is not supported by WebLLM",
        ),
      );
    }

    // Convert messages to WebLLM format
    const messages = convertToWebLLMMessages(prompt);

    // Build request options
    const requestOptions: any = {
      messages,
      temperature,
      max_tokens: maxOutputTokens,
      top_p: topP,
      seed,
    };

    // Handle response format
    if (responseFormat?.type === "json") {
      requestOptions.response_format = {
        type: "json_object",
        ...(responseFormat.schema && {
          schema: JSON.stringify(responseFormat.schema),
        }),
      };
    }

    return {
      messages,
      warnings,
      requestOptions,
      functionTools,
    };
  }

  /**
   * Generates a complete text response using WebLLM
   * @param options
   * @returns Promise resolving to the generated content with finish reason, usage stats, and any warnings
   * @throws {LoadSettingError} When WebLLM is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const converted = this.getArgs(options);
    const { messages, warnings, requestOptions, functionTools } = converted;

    // Extract system prompt and build tool calling prompt
    const {
      systemPrompt: originalSystemPrompt,
      messages: messagesWithoutSystem,
    } = extractSystemPrompt(messages);

    const systemPrompt = buildJsonToolSystemPrompt(
      originalSystemPrompt,
      functionTools,
      {
        allowParallelToolCalls: false,
      },
    );

    // Prepend system prompt to messages
    const promptMessages = prependSystemPromptToMessages(
      messagesWithoutSystem,
      systemPrompt,
    );

    const engine = await this.getEngine();

    const abortHandler = async () => {
      await engine.interruptGenerate();
    };

    if (options.abortSignal) {
      options.abortSignal.addEventListener("abort", abortHandler);
    }

    try {
      const response = await engine.chat.completions.create({
        ...requestOptions,
        messages: promptMessages,
        stream: false,
        ...(options.abortSignal &&
          !this.config.options.worker && { signal: options.abortSignal }),
      });

      const choice = response.choices[0];
      if (!choice) {
        throw new Error("No response choice returned from WebLLM");
      }

      const rawResponse = choice.message.content || "";

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
              total: response.usage?.prompt_tokens,
              noCache: undefined,
              cacheRead: undefined,
              cacheWrite: undefined,
            },
            outputTokens: {
              total: response.usage?.completion_tokens,
              text: undefined,
              reasoning: undefined,
            },
          },
          request: { body: { messages: promptMessages, ...requestOptions } },
          warnings,
        };
      }

      const content: LanguageModelV3Content[] = [
        {
          type: "text",
          text: textContent || rawResponse,
        },
      ];

      let finishReason: LanguageModelV3FinishReason = {
        unified: "stop",
        raw: choice.finish_reason,
      };
      if (choice.finish_reason === "abort") {
        finishReason = { unified: "other", raw: choice.finish_reason };
      }

      return {
        content,
        finishReason,
        usage: {
          inputTokens: {
            total: response.usage?.prompt_tokens,
            noCache: undefined,
            cacheRead: undefined,
            cacheWrite: undefined,
          },
          outputTokens: {
            total: response.usage?.completion_tokens,
            text: undefined,
            reasoning: undefined,
          },
          raw: {
            total: response.usage?.total_tokens,
          },
        },
        request: { body: { messages: promptMessages, ...requestOptions } },
        warnings,
      };
    } catch (error) {
      throw new Error(
        `WebLLM generation failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      );
    } finally {
      if (options.abortSignal) {
        options.abortSignal.removeEventListener("abort", abortHandler);
      }
    }
  }

  /**
   * Check the availability of the WebLLM model.
   * Note: On mobile devices with a worker, WebGPU detection is skipped since it
   * can't be done reliably. The actual availability will be determined at init.
   * @returns Promise resolving to "unavailable", "available", or "downloadable"
   */
  public async availability(): Promise<Availability> {
    if (this.isInitialized) {
      return "available";
    }

    // Skip on mobile if using a worker, since detecting GPU is unreliable.
    // Let worker initialization handle it and return an error if unavailable.
    if (this.config.options.worker && isMobile()) {
      return "downloadable";
    }

    const supported = checkWebGPU();
    return supported ? "downloadable" : "unavailable";
  }

  /**
   * Creates an engine session with download progress monitoring.
   *
   * @example
   * ```typescript
   * const engine = await model.createSessionWithProgress(
   *   (progress) => {
   *     console.log(`Download progress: ${Math.round(progress.loaded * 100)}%`);
   *   }
   * );
   * ```
   *
   * @param onInitProgress Optional callback receiving progress reports during model download
   * @returns Promise resolving to a configured WebLLM engine
   * @throws {LoadSettingError} When WebLLM is not available or model is unavailable
   */
  public async createSessionWithProgress(
    onInitProgress?: (progress: InitProgressReport) => void,
  ): Promise<MLCEngineInterface> {
    return this.getEngine(undefined, onInitProgress);
  }

  /**
   * Generates a streaming text response using WebLLM
   * @param options
   * @returns Promise resolving to a readable stream of text chunks and request metadata
   * @throws {LoadSettingError} When WebLLM is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const converted = this.getArgs(options);
    const { messages, warnings, requestOptions, functionTools } = converted;

    // Extract system prompt and build tool calling prompt
    const {
      systemPrompt: originalSystemPrompt,
      messages: messagesWithoutSystem,
    } = extractSystemPrompt(messages);

    const systemPrompt = buildJsonToolSystemPrompt(
      originalSystemPrompt,
      functionTools,
      {
        allowParallelToolCalls: false,
      },
    );

    // Prepend system prompt to messages
    const promptMessages = prependSystemPromptToMessages(
      messagesWithoutSystem,
      systemPrompt,
    );

    const engine = await this.getEngine();
    const useWorker = this.config.options.worker != null;

    const abortHandler = async () => {
      await engine.interruptGenerate();
    };

    if (options.abortSignal) {
      options.abortSignal.addEventListener("abort", abortHandler);
    }

    const textId = "text-0";

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({
          type: "stream-start",
          warnings,
        });

        let textStarted = false;
        let finished = false;

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

        const finishStream = (
          finishReason: LanguageModelV3FinishReason,
          usage?: {
            prompt_tokens?: number;
            completion_tokens?: number;
            total_tokens?: number;
          },
        ) => {
          if (finished) return;
          finished = true;
          emitTextEndIfNeeded();
          controller.enqueue({
            type: "finish",
            finishReason,
            usage: {
              inputTokens: {
                total: usage?.prompt_tokens,
                noCache: undefined,
                cacheRead: undefined,
                cacheWrite: undefined,
              },
              outputTokens: {
                total: usage?.completion_tokens,
                text: undefined,
                reasoning: undefined,
              },
              raw: {
                total: usage?.total_tokens,
              },
            },
          });
          controller.close();
        };

        try {
          const streamingRequest: ChatCompletionRequestStreaming = {
            ...requestOptions,
            messages: promptMessages,
            stream: true,
            stream_options: { include_usage: true },
            ...(options.abortSignal &&
              !useWorker && { signal: options.abortSignal }),
          };

          const response =
            await engine.chat.completions.create(streamingRequest);

          // Use ToolCallFenceDetector for real-time streaming
          const fenceDetector = new ToolCallFenceDetector();
          let accumulatedText = "";

          // Streaming tool call state
          let currentToolCallId: string | null = null;
          let toolInputStartEmitted = false;
          let accumulatedFenceContent = "";
          let streamedArgumentsLength = 0;
          let insideFence = false;

          for await (const chunk of response) {
            const choice = chunk.choices[0];
            if (!choice) continue;

            if (choice.delta.content) {
              const delta = choice.delta.content;
              accumulatedText += delta;

              // Add chunk to detector
              fenceDetector.addChunk(delta);

              // Process buffer using streaming detection
              while (fenceDetector.hasContent()) {
                const wasInsideFence = insideFence;
                const result = fenceDetector.detectStreamingFence();
                insideFence = result.inFence;

                let madeProgress = false;

                if (!wasInsideFence && result.inFence) {
                  if (result.safeContent) {
                    emitTextDelta(result.safeContent);
                    madeProgress = true;
                  }

                  currentToolCallId = `call_${Date.now()}_${Math.random()
                    .toString(36)
                    .slice(2, 9)}`;
                  toolInputStartEmitted = false;
                  accumulatedFenceContent = "";
                  streamedArgumentsLength = 0;
                  insideFence = true;

                  continue;
                }

                if (result.completeFence) {
                  madeProgress = true;
                  if (result.safeContent) {
                    accumulatedFenceContent += result.safeContent;
                  }

                  if (toolInputStartEmitted && currentToolCallId) {
                    const argsContent = extractArgumentsContent(
                      accumulatedFenceContent,
                    );
                    if (argsContent.length > streamedArgumentsLength) {
                      const delta = argsContent.slice(streamedArgumentsLength);
                      streamedArgumentsLength = argsContent.length;
                      if (delta.length > 0) {
                        controller.enqueue({
                          type: "tool-input-delta",
                          id: currentToolCallId,
                          delta,
                        });
                      }
                    }
                  }

                  const parsed = parseJsonFunctionCalls(result.completeFence);
                  const parsedToolCalls = parsed.toolCalls;
                  const selectedToolCalls = parsedToolCalls.slice(0, 1);

                  if (selectedToolCalls.length === 0) {
                    emitTextDelta(result.completeFence);
                    if (result.textAfterFence) {
                      emitTextDelta(result.textAfterFence);
                    }

                    currentToolCallId = null;
                    toolInputStartEmitted = false;
                    accumulatedFenceContent = "";
                    streamedArgumentsLength = 0;
                    insideFence = false;
                    continue;
                  }

                  if (selectedToolCalls.length > 0 && currentToolCallId) {
                    selectedToolCalls[0].toolCallId = currentToolCallId;
                  }

                  for (const [index, call] of selectedToolCalls.entries()) {
                    const toolCallId =
                      index === 0 && currentToolCallId
                        ? currentToolCallId
                        : call.toolCallId;
                    const toolName = call.toolName;
                    const argsJson = JSON.stringify(call.args ?? {});

                    if (toolCallId === currentToolCallId) {
                      if (!toolInputStartEmitted) {
                        controller.enqueue({
                          type: "tool-input-start",
                          id: toolCallId,
                          toolName,
                        });
                        toolInputStartEmitted = true;
                      }

                      const argsContent = extractArgumentsContent(
                        accumulatedFenceContent,
                      );
                      if (argsContent.length > streamedArgumentsLength) {
                        const delta = argsContent.slice(
                          streamedArgumentsLength,
                        );
                        streamedArgumentsLength = argsContent.length;
                        if (delta.length > 0) {
                          controller.enqueue({
                            type: "tool-input-delta",
                            id: toolCallId,
                            delta,
                          });
                        }
                      }
                    } else {
                      controller.enqueue({
                        type: "tool-input-start",
                        id: toolCallId,
                        toolName,
                      });
                      if (argsJson.length > 0) {
                        controller.enqueue({
                          type: "tool-input-delta",
                          id: toolCallId,
                          delta: argsJson,
                        });
                      }
                    }

                    controller.enqueue({
                      type: "tool-input-end",
                      id: toolCallId,
                    });
                    controller.enqueue({
                      type: "tool-call",
                      toolCallId,
                      toolName,
                      input: argsJson,
                      providerExecuted: false,
                    });
                  }

                  if (result.textAfterFence) {
                    emitTextDelta(result.textAfterFence);
                  }

                  madeProgress = true;

                  currentToolCallId = null;
                  toolInputStartEmitted = false;
                  accumulatedFenceContent = "";
                  streamedArgumentsLength = 0;
                  insideFence = false;
                  continue;
                }

                if (insideFence) {
                  if (result.safeContent) {
                    accumulatedFenceContent += result.safeContent;
                    madeProgress = true;

                    const toolName = extractToolName(accumulatedFenceContent);
                    if (
                      toolName &&
                      !toolInputStartEmitted &&
                      currentToolCallId
                    ) {
                      controller.enqueue({
                        type: "tool-input-start",
                        id: currentToolCallId,
                        toolName,
                      });
                      toolInputStartEmitted = true;
                    }

                    if (toolInputStartEmitted && currentToolCallId) {
                      const argsContent = extractArgumentsContent(
                        accumulatedFenceContent,
                      );
                      if (argsContent.length > streamedArgumentsLength) {
                        const delta = argsContent.slice(
                          streamedArgumentsLength,
                        );
                        streamedArgumentsLength = argsContent.length;
                        if (delta.length > 0) {
                          controller.enqueue({
                            type: "tool-input-delta",
                            id: currentToolCallId,
                            delta,
                          });
                        }
                      }
                    }
                  }

                  continue;
                }

                if (!insideFence && result.safeContent) {
                  emitTextDelta(result.safeContent);
                  madeProgress = true;
                }

                if (!madeProgress) {
                  break;
                }
              }
            }

            if (choice.finish_reason) {
              // Emit any remaining buffer content
              if (fenceDetector.hasContent()) {
                emitTextDelta(fenceDetector.getBuffer());
                fenceDetector.clearBuffer();
              }

              let finishReason: LanguageModelV3FinishReason = {
                unified: "stop",
                raw: "stop",
              };
              if (choice.finish_reason === "abort") {
                finishReason = { unified: "other", raw: "abort" };
              } else {
                // Check if we detected any tool calls
                const { toolCalls } = parseJsonFunctionCalls(accumulatedText);
                if (toolCalls.length > 0) {
                  finishReason = { unified: "tool-calls", raw: "tool-calls" };
                }
              }

              finishStream(finishReason, chunk.usage);
            }
          }

          if (!finished) {
            finishStream({ unified: "stop", raw: "stop" });
          }
        } catch (error) {
          // Propagate all other errors.
          controller.error(error);
        } finally {
          if (options.abortSignal) {
            options.abortSignal.removeEventListener("abort", abortHandler);
          }
          if (!finished) {
            controller.close();
          }
        }
      },
    });

    return {
      stream,
      request: { body: { messages: promptMessages, ...requestOptions } },
    };
  }
}
