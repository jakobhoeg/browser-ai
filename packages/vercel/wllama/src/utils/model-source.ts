import type { HuggingFaceParams } from "@wllama/wllama/esm/index.js";

export type WllamaModelSource =
  | { type: "hf"; params: HuggingFaceParams }
  | { type: "url"; url: string };

/**
 * Parse a model id into a wllama model source.
 *
 * - `https://example.com/model.gguf` — direct URL (split models: first shard)
 * - `hf:owner/repo/path/to/model.gguf` — file inside a Hugging Face repo
 */
export function parseModelId(modelId: string): WllamaModelSource {
  if (/^https?:\/\//i.test(modelId)) {
    return { type: "url", url: modelId };
  }

  const match = /^hf:([^/]+\/[^/]+)\/(.+\.gguf)$/i.exec(modelId);
  if (!match) {
    throw new Error(
      `Invalid wllama model id "${modelId}". Expected "hf:owner/repo/file.gguf" or an https URL to a .gguf file.`,
    );
  }
  return { type: "hf", params: { repo: match[1], file: match[2] } };
}
