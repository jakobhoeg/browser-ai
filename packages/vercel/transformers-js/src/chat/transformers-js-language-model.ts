import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  SharedV3Warning,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3ProviderTool,
  LanguageModelV3StreamPart,
  LoadSettingError,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoProcessor,
  AutoModelForVision2Seq,
  StoppingCriteria,
  type PretrainedModelOptions,
  type ProgressInfo,
} from "@huggingface/transformers";
import { convertToTransformersMessages } from "./convert-to-transformers-message";
import type { TransformersMessage } from "./convert-to-transformers-message";
import type {
  ModelInstance,
  GenerationOptions,
} from "./transformers-js-worker-types";
import {
  parseJsonFunctionCalls,
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  isFunctionTool,
  ToolCallFenceDetector,
  type ParsedToolCall,
  type ToolDefinition,
} from "@browser-ai/shared";
import {
  createMainThreadGenerationStream,
  createWorkerGenerationStream,
} from "./generation-stream";

declare global {
  interface Navigator {
    gpu?: unknown;
  }
}

export type TransformersJSModelId = string;

export interface TransformersJSModelSettings extends Pick<
  PretrainedModelOptions,
  "device" | "dtype"
> {
  /**
   * Progress callback for model initialization
   */
  initProgressCallback?: (progress: { progress: number }) => void;
  /**
   * Raw progress callback from Transformers.js
   */
  rawInitProgressCallback?: (progress: ProgressInfo) => void;
  /**
   * Whether this is a vision model
   * @default false
   */
  isVisionModel?: boolean;
  /**
   * Optional Web Worker to run the model off the main thread
   */
  worker?: Worker;
}

/**
 * Check if we're running in a browser environment
 */
export function isBrowserEnvironment(): boolean {
  return typeof window !== "undefined";
}

/**
 * Check if we're running in a server environment (Node.js)
 */
export function isServerEnvironment(): boolean {
  return typeof window === "undefined" && typeof process !== "undefined";
}

/**
 * Check if the browser supports TransformersJS with optimal performance
 * Returns true if the browser has WebGPU or WebAssembly support
 * @returns true if the browser supports TransformersJS, false otherwise
 */
export function doesBrowserSupportTransformersJS(): boolean {
  if (!isBrowserEnvironment()) {
    return false;
  }

  // Check for WebGPU support for better performance
  if (typeof navigator !== "undefined" && navigator.gpu) {
    return true;
  }

  // Check for WebAssembly support as fallback
  if (typeof WebAssembly !== "undefined") {
    return true;
  }

  return false;
}

// Simplified config - just extend the settings with modelId
interface ModelConfig extends TransformersJSModelSettings {
  modelId: TransformersJSModelId;
}

class InterruptableStoppingCriteria extends StoppingCriteria {
  interrupted = false;

  interrupt() {
    this.interrupted = true;
  }

  reset() {
    this.interrupted = false;
  }

  _call(input_ids: number[][], scores: number[][]): boolean[] {
    return new Array(input_ids.length).fill(this.interrupted);
  }
}

/**
 * Extract tool name from partial fence content for early emission
 * This allows us to emit tool-input-start as soon as we know the tool name
 * Expects JSON format: {"name":"toolName"
 */
function extractToolName(content: string): string | null {
  // For JSON mode: {"name":"toolName"
  const jsonMatch = content.match(/\{\s*"name"\s*:\s*"([^"]+)"/);
  if (jsonMatch) {
    return jsonMatch[1];
  }
  return null;
}

/**
 * Extract the argument section from a streaming tool call fence.
 * Returns the substring after `"arguments":` (best-effort for partial JSON).
 */
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

export class TransformersJSLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly modelId: TransformersJSModelId;
  readonly provider = "transformers-js";

  private readonly config: ModelConfig;
  private modelInstance?: ModelInstance;
  private isInitialized = false;
  private initializationPromise?: Promise<void>;
  private stoppingCriteria = new InterruptableStoppingCriteria();
  private workerReady = false;

  constructor(
    modelId: TransformersJSModelId,
    options: TransformersJSModelSettings = {},
  ) {
    this.modelId = modelId;
    this.config = {
      modelId,
      device: "auto",
      dtype: "auto",
      isVisionModel: false,
      ...options,
    };
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    // TransformersJS doesn't support URLs natively
  };

  private async getSession(
    onInitProgress?: (progress: { progress: number }) => void,
  ): Promise<ModelInstance> {
    if (this.modelInstance && this.isInitialized) {
      return this.modelInstance;
    }

    if (this.initializationPromise) {
      await this.initializationPromise;
      if (this.modelInstance) {
        return this.modelInstance;
      }
    }

    this.initializationPromise = this._initializeModel(onInitProgress);
    await this.initializationPromise;

    if (!this.modelInstance) {
      throw new LoadSettingError({
        message: "Model initialization failed",
      });
    }

    return this.modelInstance;
  }

  private async _initializeModel(
    onInitProgress?: (progress: { progress: number }) => void,
  ): Promise<void> {
    try {
      const { isVisionModel, device, dtype } = this.config;
      const progress_callback = this.createProgressTracker(onInitProgress);

      // Set device based on environment
      const resolvedDevice = this.resolveDevice(
        device as string,
      ) as PretrainedModelOptions["device"];
      const resolvedDtype = this.resolveDtype(
        dtype as string,
      ) as PretrainedModelOptions["dtype"];

      // Create model instance based on type
      if (isVisionModel) {
        const [processor, model] = await Promise.all([
          AutoProcessor.from_pretrained(this.modelId, { progress_callback }),
          AutoModelForVision2Seq.from_pretrained(this.modelId, {
            dtype: resolvedDtype,
            device: resolvedDevice,
            progress_callback,
          }),
        ]);
        this.modelInstance = [processor, model];
      } else {
        const [tokenizer, model] = await Promise.all([
          AutoTokenizer.from_pretrained(this.modelId, {
            legacy: true,
            progress_callback,
          }),
          AutoModelForCausalLM.from_pretrained(this.modelId, {
            dtype: resolvedDtype,
            device: resolvedDevice,
            progress_callback,
          }),
        ]);
        this.modelInstance = [tokenizer, model];

        // Warm up text models (skip in server environment to reduce initialization time)
        if (isBrowserEnvironment()) {
          const dummyInputs = tokenizer("Hello");
          await model.generate({ ...dummyInputs, max_new_tokens: 1 });
        }
      }

      onInitProgress?.({ progress: 1.0 });
      this.isInitialized = true;
    } catch (error) {
      this.modelInstance = undefined;
      this.isInitialized = false;
      this.initializationPromise = undefined;

      throw new LoadSettingError({
        message: `Failed to initialize TransformersJS model: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  }

  private resolveDevice(device?: string): string {
    if (device && device !== "auto") {
      return device;
    }

    if (isServerEnvironment()) {
      // In server environment, prefer CPU unless explicitly set
      return "cpu";
    }

    // In browser environment, auto-detect WebGPU support
    if (
      isBrowserEnvironment() &&
      typeof navigator !== "undefined" &&
      navigator.gpu
    ) {
      return "webgpu";
    }

    return "cpu";
  }

  private resolveDtype(dtype?: string): string {
    if (dtype && dtype !== "auto") {
      return dtype;
    }

    return "auto";
  }

  private createProgressTracker(
    onInitProgress?: (progress: { progress: number }) => void,
  ) {
    const fileProgress = new Map<string, { loaded: number; total: number }>();

    return (p: ProgressInfo) => {
      // Pass through raw progress
      this.config.rawInitProgressCallback?.(p);

      if (!onInitProgress) return;

      // Type guard to check if p has file property
      const progressWithFile = p as ProgressInfo & {
        file?: string;
        loaded?: number;
        total?: number;
      };
      const file = progressWithFile.file;

      if (!file) return;

      if (p.status === "progress" && file) {
        fileProgress.set(file, {
          loaded: progressWithFile.loaded || 0,
          total: progressWithFile.total || 0,
        });
      } else if (p.status === "done" && file) {
        const prev = fileProgress.get(file);
        if (prev?.total) {
          fileProgress.set(file, { loaded: prev.total, total: prev.total });
        }
      }

      // Calculate overall progress
      let totalLoaded = 0;
      let totalBytes = 0;
      for (const { loaded, total } of fileProgress.values()) {
        if (total > 0) {
          totalLoaded += loaded;
          totalBytes += total;
        }
      }

      if (totalBytes > 0) {
        onInitProgress({ progress: Math.min(1, totalLoaded / totalBytes) });
      }
    };
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
  }: Parameters<LanguageModelV3["doGenerate"]>[0]): {
    messages: TransformersMessage[];
    warnings: SharedV3Warning[];
    generationOptions: GenerationOptions;
    functionTools: ToolDefinition[];
  } {
    const warnings: SharedV3Warning[] = [];
    // Filter and warn about unsupported tools
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
          "Only function tools are supported by TransformersJS",
        ),
      );
    }

    // Add warnings for unsupported settings
    if (frequencyPenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "frequencyPenalty",
          "Frequency penalty is not supported by TransformersJS",
        ),
      );
    }

    if (presencePenalty != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "presencePenalty",
          "Presence penalty is not supported by TransformersJS",
        ),
      );
    }

    if (stopSequences != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "stopSequences",
          "Stop sequences are not supported by TransformersJS",
        ),
      );
    }

    if (responseFormat?.type === "json") {
      warnings.push(
        createUnsupportedSettingWarning(
          "responseFormat",
          "JSON response format is not supported by TransformersJS",
        ),
      );
    }

    if (seed != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "seed",
          "Seed is not supported by TransformersJS",
        ),
      );
    }

    if (toolChoice != null) {
      warnings.push(
        createUnsupportedSettingWarning(
          "toolChoice",
          "toolChoice is not supported by TransformersJS",
        ),
      );
    }

    // Convert messages to TransformersJS format
    const messages = convertToTransformersMessages(
      prompt,
      this.config.isVisionModel,
    );

    const generationOptions: GenerationOptions = {
      max_new_tokens: maxOutputTokens || 32768,
      temperature: temperature || 0.7,
      top_p: topP,
      top_k: topK,
      do_sample: temperature !== undefined && temperature > 0,
    };

    return {
      messages,
      warnings,
      generationOptions,
      functionTools,
    };
  }

  /**
   * Check the availability of the TransformersJS model
   */
  public async availability(): Promise<
    "unavailable" | "downloadable" | "available"
  > {
    // If using a worker (browser only), reflect worker readiness instead of main-thread state
    if (this.config.worker && isBrowserEnvironment()) {
      return this.workerReady ? "available" : "downloadable";
    }

    // In server environment, workers are not used
    if (isServerEnvironment() && this.config.worker) {
      // Ignore worker config on server and use main thread
    }

    if (this.isInitialized) {
      return "available";
    }

    return "downloadable";
  }

  /**
   * Creates a session with download progress monitoring
   */
  public async createSessionWithProgress(
    onDownloadProgress?: (progress: { progress: number }) => void,
  ): Promise<TransformersJSLanguageModel> {
    // If a worker is provided and we're in browser environment, initialize the worker
    // (and forward progress) instead of initializing the model on the main thread
    // to avoid double-initialization/downloads.
    if (this.config.worker && isBrowserEnvironment()) {
      await this.initializeWorker(onDownloadProgress);
      return this;
    }

    // In server environment or when no worker is provided, use main thread
    await this._initializeModel(onDownloadProgress);
    return this;
  }

  /**
   * Generates a complete text response using TransformersJS
   */
  public async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const { messages, warnings, generationOptions, functionTools } =
      this.getArgs(options);

    const useWorker = this.config.worker && isBrowserEnvironment();

    // Initialize worker if needed
    if (useWorker) {
      await this.initializeWorker();
    }

    try {
      // Create the appropriate generation stream
      const generationStream = useWorker
        ? createWorkerGenerationStream({
            worker: this.config.worker!,
            messages,
            generationOptions,
            tools: functionTools,
            abortSignal: options.abortSignal,
          })
        : createMainThreadGenerationStream({
            modelInstance: await this.getSession(
              this.config.initProgressCallback,
            ),
            messages,
            generationOptions,
            tools: functionTools,
            isVisionModel: this.config.isVisionModel,
            stoppingCriteria: this.stoppingCriteria,
            abortSignal: options.abortSignal,
          });

      // Collect all generated text
      let generatedText = "";
      let lastUsage: { inputTokens?: number; outputTokens?: number } = {};
      let workerToolCalls: ParsedToolCall[] = [];

      for await (const event of generationStream) {
        if (event.type === "delta") {
          generatedText += event.delta;
        } else if (event.type === "complete") {
          lastUsage = event.usage || {};
          if (event.toolCalls) {
            workerToolCalls = event.toolCalls;
          }
        }
      }

      // Parse for tool calls - prefer worker-parsed ones if available
      const { toolCalls: parsedToolCalls, textContent } =
        parseJsonFunctionCalls(generatedText);
      const toolCalls =
        workerToolCalls.length > 0 ? workerToolCalls : parsedToolCalls;

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
          });
        }

        return {
          content: parts,
          finishReason: { unified: "tool-calls", raw: "tool-calls" },
          usage: {
            inputTokens: {
              total: lastUsage.inputTokens,
              noCache: undefined,
              cacheRead: undefined,
              cacheWrite: undefined,
            },
            outputTokens: {
              total: lastUsage.outputTokens,
              text: undefined,
              reasoning: undefined,
            },
          },
          request: { body: { messages, ...generationOptions } },
          warnings,
        };
      }

      const content: LanguageModelV3Content[] = [
        {
          type: "text",
          text: textContent || generatedText,
        },
      ];

      return {
        content,
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: {
            total: lastUsage.inputTokens,
            noCache: undefined,
            cacheRead: undefined,
            cacheWrite: undefined,
          },
          outputTokens: {
            total: lastUsage.outputTokens,
            text: undefined,
            reasoning: undefined,
          },
        },
        request: { body: { messages, ...generationOptions } },
        warnings,
      };
    } catch (error) {
      throw new Error(
        `TransformersJS generation failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      );
    }
  }

  private async initializeWorker(
    onInitProgress?: (progress: { progress: number }) => void,
  ): Promise<void> {
    if (!this.config.worker) return;

    // If already ready, optionally emit completion progress
    if (this.workerReady) {
      if (onInitProgress) onInitProgress({ progress: 1 });
      return;
    }

    const worker = this.config.worker;

    await new Promise<void>((resolve, reject) => {
      const trackProgress = this.createProgressTracker(onInitProgress);

      const onMessage = (e: MessageEvent) => {
        const msg = e.data;
        if (!msg) return;

        // Forward raw download progress events coming from @huggingface/transformers running in the worker
        if (msg && typeof msg === "object" && "status" in msg) {
          if (msg.status === "ready") {
            worker.removeEventListener("message", onMessage);
            this.workerReady = true;
            if (onInitProgress) onInitProgress({ progress: 1 });
            resolve();
            return;
          }
          if (msg.status === "error") {
            worker.removeEventListener("message", onMessage);
            reject(
              new Error(String(msg.data || "Worker initialization failed")),
            );
            return;
          }

          // Only track file-related messages (raw ProgressInfo events)
          const msgWithFile = msg as ProgressInfo & { file?: string };
          if (msgWithFile.file) trackProgress(msg as ProgressInfo);
        }
      };

      worker.addEventListener("message", onMessage);
      worker.postMessage({
        type: "load",
        data: {
          modelId: this.modelId,
          dtype: this.config.dtype,
          device: this.config.device,
          isVisionModel: this.config.isVisionModel,
        },
      });
    });
  }

  /**
   * Generates a streaming text response using TransformersJS
   */
  public async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const { messages, warnings, generationOptions, functionTools } =
      this.getArgs(options);

    const useWorker = this.config.worker && isBrowserEnvironment();

    // Initialize worker if needed
    if (useWorker) {
      await this.initializeWorker();
    }

    const self = this;
    const textId = "text-0";

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: "stream-start", warnings });

        let textStarted = false;

        const emitTextDelta = (delta: string) => {
          if (!delta) return;
          if (!textStarted) {
            controller.enqueue({ type: "text-start", id: textId });
            textStarted = true;
          }
          controller.enqueue({ type: "text-delta", id: textId, delta });
        };

        try {
          // Create the appropriate generation stream
          const generationStream = useWorker
            ? createWorkerGenerationStream({
                worker: self.config.worker!,
                messages,
                generationOptions,
                tools: functionTools,
                abortSignal: options.abortSignal,
              })
            : createMainThreadGenerationStream({
                modelInstance: await self.getSession(
                  self.config.initProgressCallback,
                ),
                messages,
                generationOptions,
                tools: functionTools,
                isVisionModel: self.config.isVisionModel,
                stoppingCriteria: self.stoppingCriteria,
                abortSignal: options.abortSignal,
              });

          const fenceDetector = new ToolCallFenceDetector();
          let accumulatedText = "";

          let currentToolCallId: string | null = null;
          let toolInputStartEmitted = false;
          let accumulatedFenceContent = "";
          let streamedArgumentsLength = 0;
          let insideFence = false;
          let toolCallDetected = false;

          let lastUsage: { inputTokens?: number; outputTokens?: number } = {};

          for await (const event of generationStream) {
            if (event.type === "delta") {
              accumulatedText += event.delta;
              fenceDetector.addChunk(event.delta);

              // Process buffer using streaming detection
              while (fenceDetector.hasContent() && !toolCallDetected) {
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
                    });
                  }

                  if (result.textAfterFence) {
                    emitTextDelta(result.textAfterFence);
                  }

                  madeProgress = true;
                  toolCallDetected = true;

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
            } else if (event.type === "complete") {
              lastUsage = event.usage || {};
            }
          }

          // Emit any remaining buffer content
          if (fenceDetector.hasContent()) {
            emitTextDelta(fenceDetector.getBuffer());
            fenceDetector.clearBuffer();
          }

          if (textStarted) {
            controller.enqueue({ type: "text-end", id: textId });
          }

          // Determine finish reason
          const { toolCalls } = parseJsonFunctionCalls(accumulatedText);
          const finishReason: LanguageModelV3FinishReason =
            toolCalls.length > 0
              ? { unified: "tool-calls", raw: "tool-calls" }
              : { unified: "stop", raw: "stop" };

          controller.enqueue({
            type: "finish",
            finishReason,
            usage: {
              inputTokens: {
                total: lastUsage.inputTokens,
                noCache: undefined,
                cacheRead: undefined,
                cacheWrite: undefined,
              },
              outputTokens: {
                total: lastUsage.outputTokens,
                text: undefined,
                reasoning: undefined,
              },
            },
          });
          controller.close();
        } catch (error) {
          controller.enqueue({ type: "error", error });
          controller.close();
        }
      },
    });

    return {
      stream,
      request: { body: { messages, ...generationOptions } },
    };
  }
}
