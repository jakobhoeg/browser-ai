export {
  LiteRTLMLanguageModel,
  doesBrowserSupportLiteRTLM,
} from "./chat/litert-lm-language-model";
export type {
  LiteRTLMModelId,
  LiteRTLMSettings,
} from "./chat/litert-lm-language-model";

export type { Availability } from "./types";

export { liteRTLM, createLiteRTLM } from "./litert-lm-provider";
export type { LiteRTLMProvider } from "./litert-lm-provider";

/**
 * Models supported by the LiteRT-LM Web SDK (text-in / text-out, WebGPU).
 * These are direct URLs to the `.litertlm` artifacts on Hugging Face; pass
 * them (or any `.litertlm` URL/path/Blob) as the model id.
 */
export const GEMMA_4_E2B_WEB =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm";

export const GEMMA_4_E4B_WEB =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm";
