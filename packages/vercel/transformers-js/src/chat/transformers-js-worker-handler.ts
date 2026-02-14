import {
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoProcessor,
  AutoModelForVision2Seq,
  TextStreamer,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  load_image,
  type ProgressInfo,
} from "@huggingface/transformers";
import { decodeGeneratedText } from "./decode-utils";
import {
  parseJsonFunctionCalls,
  ToolCallFenceDetector,
  type ToolDefinition,
  type ParsedToolCall,
} from "@browser-ai/shared";
import { convertToolsToHuggingFaceFormat } from "./convert-tools";
import type {
  WorkerMessage,
  WorkerGlobalScope,
  ModelInstance,
  WorkerLoadOptions,
  GenerationOptions,
  WorkerGenerateData,
  WorkerLoadData,
} from "./transformers-js-worker-types";
import type { PretrainedModelOptions } from "@huggingface/transformers";

declare const self: WorkerGlobalScope;

class ModelManager {
  private static configs = new Map<string, WorkerLoadOptions>();
  private static instances = new Map<string, Promise<ModelInstance>>();

  static configure(key: string, options: WorkerLoadOptions) {
    this.configs.set(key, options);
  }

  static async getInstance(
    key: string,
    progressCallback?: (progress: ProgressInfo) => void,
  ): Promise<ModelInstance> {
    const cached = this.instances.get(key);
    if (cached) return cached;

    const config = this.configs.get(key);
    if (!config || !config.modelId) {
      throw new Error(`No configuration found for key: ${key}`);
    }

    const {
      modelId,
      dtype = "auto",
      device = "auto",
      use_external_data_format,
      isVisionModel = false,
    } = config;

    const instancePromise = isVisionModel
      ? this.createVisionModel(modelId, {
          dtype,
          device,
          use_external_data_format,
          progressCallback,
        })
      : this.createTextModel(modelId, {
          dtype,
          device,
          use_external_data_format,
          progressCallback,
        });

    this.instances.set(key, instancePromise);
    return instancePromise;
  }

  private static async createTextModel(
    modelId: string,
    options: {
      dtype?: PretrainedModelOptions["dtype"];
      device?: PretrainedModelOptions["device"];
      use_external_data_format?: boolean;
      progressCallback?: (progress: ProgressInfo) => void;
    },
  ): Promise<ModelInstance> {
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(modelId, {
        progress_callback: options.progressCallback,
        legacy: true,
      }),
      AutoModelForCausalLM.from_pretrained(modelId, {
        dtype: options.dtype,
        device: options.device,
        ...(options.use_external_data_format !== undefined
          ? { use_external_data_format: options.use_external_data_format }
          : {}),
        progress_callback: options.progressCallback,
      }),
    ]);
    return [tokenizer, model];
  }

  private static async createVisionModel(
    modelId: string,
    options: {
      dtype?: PretrainedModelOptions["dtype"];
      device?: PretrainedModelOptions["device"];
      use_external_data_format?: boolean;
      progressCallback?: (progress: ProgressInfo) => void;
    },
  ): Promise<ModelInstance> {
    const [processor, model] = await Promise.all([
      AutoProcessor.from_pretrained(modelId, {
        progress_callback: options.progressCallback,
      }),
      AutoModelForVision2Seq.from_pretrained(modelId, {
        dtype: options.dtype || "fp32",
        device: options.device || "webgpu",
        ...(options.use_external_data_format !== undefined
          ? { use_external_data_format: options.use_external_data_format }
          : {}),
        progress_callback: options.progressCallback,
      }),
    ]);
    return [processor, model];
  }

  static clearCache() {
    this.instances.clear();
  }
}

export class TransformersJSWorkerHandler {
  private stopping_criteria = new InterruptableStoppingCriteria();
  private isVisionModel = false;
  private currentModelKey = "default";
  private past_key_values_cache: unknown = null;
  private cachedSequenceTokenIds: number[] | null = null;

  async generate(
    messages: WorkerGenerateData[],
    generationOptions?: GenerationOptions,
    tools?: ToolDefinition[],
  ) {
    try {
      const modelInstance = await ModelManager.getInstance(
        this.currentModelKey,
      );
      await this.runGeneration(
        modelInstance,
        messages,
        generationOptions,
        tools,
      );
    } catch (error) {
      this.sendError(error instanceof Error ? error.message : String(error));
    }
  }

  private async runGeneration(
    modelInstance: ModelInstance,
    messages: WorkerGenerateData[],
    userGenerationOptions?: GenerationOptions,
    tools?: ToolDefinition[],
  ) {
    const [processor, model] = modelInstance;
    const isVision = this.isVisionModel;

    const hfTools =
      tools && tools.length > 0
        ? convertToolsToHuggingFaceFormat(tools)
        : undefined;

    const processedMessages = messages;

    // Prepare inputs based on model type
    // Using 'any' here as transformers.js returns various formats depending on model type
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let inputs: any;
    if (isVision) {
      // For vision models, use last message and extract images
      const lastMessages = processedMessages.slice(-1);
      const images = await Promise.all(
        lastMessages
          .map((x) => x.content)
          .flat(Infinity)
          .filter(
            (msg): msg is { type: string; image: string } =>
              typeof msg === "object" &&
              msg !== null &&
              "image" in msg &&
              msg.image !== undefined,
          )
          .map((msg) => load_image(msg.image)),
      );
      const text = processor.apply_chat_template(lastMessages as any, {
        add_generation_prompt: true,
        ...(hfTools ? { tools: hfTools } : {}),
      });
      inputs = await processor(text, images);
    } else {
      inputs = processor.apply_chat_template(processedMessages as any, {
        add_generation_prompt: true,
        return_dict: true,
        ...(hfTools ? { tools: hfTools } : {}),
      });
    }

    const inputTokenIds = isVision
      ? null
      : this.extractTokenIds(inputs.input_ids);
    if (
      !isVision &&
      this.past_key_values_cache !== null &&
      (!inputTokenIds || !this.canReuseCache(inputTokenIds))
    ) {
      this.clearGenerationCache();
    }

    // Setup performance tracking and tool call detection
    let startTime: number | undefined;
    let numTokens = 0;
    const fenceDetector = new ToolCallFenceDetector();
    let accumulatedText = "";
    let toolCallDetected = false;

    const token_callback = () => {
      startTime ??= performance.now();
      numTokens++;
    };
    const output_callback = (output: string) => {
      accumulatedText += output;

      if (tools && tools.length > 0 && !toolCallDetected) {
        fenceDetector.addChunk(output);
        const result = fenceDetector.detectStreamingFence();

        // If we detect a complete fence, check if it's a valid tool call
        if (result.completeFence) {
          const { toolCalls } = parseJsonFunctionCalls(result.completeFence);
          if (toolCalls.length > 0) {
            toolCallDetected = true;
            this.stopping_criteria.interrupt();
          }
        }
      }

      const tps = startTime
        ? (numTokens / (performance.now() - startTime)) * 1000
        : undefined;
      this.sendUpdate(output, tps, numTokens);
    };

    const streamer = new TextStreamer(
      isVision ? (processor as any).tokenizer : processor,
      {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: output_callback,
        token_callback_function: token_callback,
      },
    );

    const stoppingCriteriaList = new StoppingCriteriaList();
    stoppingCriteriaList.push(this.stopping_criteria);

    // Merge user generation options with defaults based on model type
    const defaultOptions = isVision
      ? {
          do_sample: false,
          repetition_penalty: 1.1,
          max_new_tokens: 1024,
        }
      : {
          do_sample: true,
          top_k: 3,
          temperature: 0.7,
          max_new_tokens: 512,
        };

    const generationOptions = {
      ...defaultOptions,
      ...userGenerationOptions,
      streamer,
      stopping_criteria: stoppingCriteriaList,
      return_dict_in_generate: true,
    };

    this.sendMessage({ status: "start" });

    const baseOptions = Object.assign({}, inputs, generationOptions);
    const withCacheOptions =
      !isVision && this.past_key_values_cache !== null
        ? Object.assign({}, baseOptions, {
            past_key_values: this.past_key_values_cache,
          })
        : baseOptions;

    let generationOutput: unknown;
    try {
      generationOutput = await model.generate(withCacheOptions);
    } catch (error) {
      // If cached prefill is rejected by runtime/model, retry once without cache.
      if (!isVision && this.past_key_values_cache !== null && numTokens === 0) {
        this.clearGenerationCache();
        generationOutput = await model.generate(baseOptions);
      } else {
        throw error;
      }
    }
    const sequences = (generationOutput as any).sequences || generationOutput;
    const inputLength = isVision
      ? 0
      : (inputTokenIds?.length ?? inputs.input_ids.data.length);

    const decoded = decodeGeneratedText(
      processor,
      sequences,
      isVision,
      inputLength,
    );

    if (!isVision) {
      this.updateGenerationCache(generationOutput);
    }

    // Parse tool calls from the complete response if tools are available
    let toolCalls: ParsedToolCall[] = [];
    if (tools && tools.length > 0) {
      const finalText = Array.isArray(decoded) ? decoded[0] : decoded;
      const parsed = parseJsonFunctionCalls(finalText);
      toolCalls = parsed.toolCalls;
    }

    self.postMessage({
      status: "complete",
      output: decoded,
      inputLength,
      numTokens,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    });
  }

  async load(options?: WorkerLoadData) {
    try {
      ModelManager.clearCache();
      this.clearGenerationCache();

      this.isVisionModel = options?.isVisionModel || false;

      const modelId =
        options?.modelId ||
        (this.isVisionModel
          ? "HuggingFaceTB/SmolVLM-256M-Instruct"
          : "HuggingFaceTB/SmolLM2-360M-Instruct");

      ModelManager.configure(this.currentModelKey, {
        ...options,
        modelId,
      });

      this.sendMessage({ status: "loading", data: "Loading model..." });

      const throttledProgress = this.createThrottledProgressCallback();

      const modelInstance = await ModelManager.getInstance(
        this.currentModelKey,
        throttledProgress,
      );

      // Warm up model (text models only)
      if (!this.isVisionModel) {
        this.sendMessage({
          status: "loading",
          data: "Compiling shaders and warming up model...",
        });
        const [tokenizer, model] = modelInstance;
        const inputs = tokenizer("a");
        await model.generate({ ...inputs, max_new_tokens: 1 });
      } else {
        this.sendMessage({
          status: "loading",
          data: "Model loaded and ready...",
        });
      }

      this.sendMessage({ status: "ready" });
    } catch (error) {
      console.error("Error in worker load:", error);
      this.sendError(error instanceof Error ? error.message : String(error));
    }
  }

  interrupt() {
    this.stopping_criteria.interrupt();
  }

  reset() {
    this.stopping_criteria.reset();
    ModelManager.clearCache();
    this.clearGenerationCache();
  }

  private clearGenerationCache() {
    this.past_key_values_cache = null;
    this.cachedSequenceTokenIds = null;
  }

  private canReuseCache(inputTokenIds: number[]): boolean {
    if (!this.cachedSequenceTokenIds) {
      return false;
    }

    if (inputTokenIds.length < this.cachedSequenceTokenIds.length) {
      return false;
    }

    for (let i = 0; i < this.cachedSequenceTokenIds.length; i++) {
      if (inputTokenIds[i] !== this.cachedSequenceTokenIds[i]) {
        return false;
      }
    }

    return true;
  }

  private updateGenerationCache(generationOutput: unknown) {
    const output = generationOutput as { past_key_values?: unknown };
    if (!output?.past_key_values) {
      this.clearGenerationCache();
      return;
    }

    const sequenceTokenIds =
      this.extractFirstSequenceTokenIds(generationOutput);
    if (!sequenceTokenIds) {
      this.clearGenerationCache();
      return;
    }

    this.past_key_values_cache = output.past_key_values;
    this.cachedSequenceTokenIds = sequenceTokenIds;
  }

  private extractFirstSequenceTokenIds(
    generationOutput: unknown,
  ): number[] | null {
    const output = generationOutput as { sequences?: unknown };
    const sequences = output?.sequences ?? generationOutput;

    if (Array.isArray(sequences)) {
      if (sequences.length === 0) {
        return null;
      }

      const first = sequences[0];
      if (typeof first === "number") {
        return (sequences as number[]).slice();
      }
      return this.extractTokenIds(first);
    }

    return this.extractTokenIds(sequences);
  }

  private extractTokenIds(value: unknown): number[] | null {
    if (!value || typeof value !== "object") {
      return null;
    }

    const maybeData = (value as { data?: unknown }).data;
    if (Array.isArray(maybeData)) {
      return maybeData.slice();
    }

    if (ArrayBuffer.isView(maybeData)) {
      if (maybeData instanceof DataView) {
        return null;
      }
      return Array.from(maybeData as unknown as ArrayLike<number>);
    }

    return null;
  }

  private sendMessage(message: {
    status: "loading" | "ready" | "start" | "complete";
    data?: string;
  }) {
    self.postMessage(message);
  }

  private sendUpdate(output: string, tps?: number, numTokens?: number) {
    self.postMessage({ status: "update", output, tps, numTokens });
  }

  private sendError(message: string) {
    self.postMessage({ status: "error", data: message });
  }

  private createThrottledProgressCallback() {
    const throttleMs = 100;
    let lastProgressTs = 0;

    return (progress: ProgressInfo) => {
      const now = performance?.now?.() ?? Date.now();
      if (progress.status === "progress") {
        if (now - lastProgressTs < throttleMs) return;
        lastProgressTs = now;
      }
      self.postMessage(progress);
    };
  }

  onmessage(e: MessageEvent<WorkerMessage>) {
    try {
      const msg = e.data;
      if (!msg) {
        this.sendError("Empty message received");
        return;
      }

      switch (msg.type) {
        case "load":
          this.load(msg.data);
          break;
        case "generate":
          this.stopping_criteria.reset();
          this.generate(msg.data, msg.generationOptions, msg.tools);
          break;
        case "interrupt":
          this.interrupt();
          break;
        case "reset":
          this.reset();
          break;
        default:
          this.sendError(
            `Unknown message type: ${(msg as { type: string }).type}`,
          );
          break;
      }
    } catch (error) {
      this.sendError(error instanceof Error ? error.message : String(error));
    }
  }
}
