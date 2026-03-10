import type {
  PreTrainedTokenizer,
  PreTrainedModel,
  PretrainedModelOptions,
  Processor,
  ProgressInfo,
} from "@huggingface/transformers";
import type { ToolDefinition, ParsedToolCall } from "@browser-ai/shared";

export interface GenerationOptions {
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  do_sample?: boolean;
  repetition_penalty?: number;
  num_beams?: number;
  early_stopping?: boolean;
}

export type { ProgressInfo } from "@huggingface/transformers";

/**
 * Message content for different worker message types
 */
export interface WorkerLoadData {
  modelId?: string;
  dtype?: PretrainedModelOptions["dtype"];
  device?: PretrainedModelOptions["device"];
  use_external_data_format?: boolean;
  isVisionModel?: boolean;
  /**
   * Name of the model class to use from @huggingface/transformers.
   * Used for vision models that require a specific class instead of AutoModelForVision2Seq.
   * @example "Qwen3_5ForConditionalGeneration"
   */
  modelArchitecture?: string;
  /**
   * Extra options forwarded to processor.apply_chat_template().
   * @example { tokenizer_kwargs: { enable_thinking: false } }
   */
  chatTemplateOptions?: Record<string, unknown>;
}

export interface WorkerGenerateData {
  role: string;
  content: string | Array<{ type: string; text?: string; image?: string }>;
}

/**
 * Message types for worker communication
 */
export type WorkerMessage =
  | { type: "load"; data?: WorkerLoadData }
  | {
      type: "generate";
      data: WorkerGenerateData[];
      generationOptions?: GenerationOptions;
      tools?: ToolDefinition[];
    }
  | { type: "interrupt" }
  | { type: "reset" };

/**
 * Worker response types
 */
export type WorkerResponse =
  | {
      status: "loading" | "ready" | "start" | "update" | "complete" | "error";
      output?: string | string[];
      data?: string;
      tps?: number;
      numTokens?: number;
      inputLength?: number;
      toolCalls?: ParsedToolCall[];
    }
  | ProgressInfo;

/**
 * Type for worker global scope
 */
export interface WorkerGlobalScope {
  postMessage(message: any): void;
  addEventListener(type: string, listener: (e: any) => void): void;
}

/**
 * Model instance types
 */
export type ModelInstance =
  | [PreTrainedTokenizer, PreTrainedModel]
  | [Processor, PreTrainedModel];

/**
 * Configuration options for worker model loading
 */
export interface WorkerLoadOptions extends Pick<
  PretrainedModelOptions,
  "dtype" | "device"
> {
  modelId?: string;
  use_external_data_format?: boolean;
  isVisionModel?: boolean;
  /**
   * Name of the model class to use from @huggingface/transformers.
   * @example "Qwen3_5ForConditionalGeneration"
   */
  modelArchitecture?: string;
  /**
   * Extra options forwarded to processor.apply_chat_template().
   * @example { tokenizer_kwargs: { enable_thinking: false } }
   */
  chatTemplateOptions?: Record<string, unknown>;
}
