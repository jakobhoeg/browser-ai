import { WorkerLoadOptions } from "@browser-ai/transformers-js";

export interface ModelConfig extends Omit<WorkerLoadOptions, "modelId"> {
  id: string;
  name: string;
  supportsWorker: boolean;
}

export const MODELS: ModelConfig[] = [
  {
    id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    name: "SmolLM2 360M",
    device: "webgpu",
    dtype: "q4",
    supportsWorker: true,
  },
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    name: "Qwen3 0.6B",
    device: "webgpu",
    dtype: "q4f16",
    supportsWorker: true,
  },
  {
    id: "onnx-community/granite-4.0-350m-ONNX-web",
    name: "Granite 4.0 350M (Tool calling)",
    device: "webgpu",
    dtype: "fp16",
    supportsWorker: true,
  },
  {
    id: "onnx-community/LFM2-1.2B-Tool-ONNX",
    name: "LFM2 1.2B-Tool",
    device: "webgpu",
    dtype: "fp16",
    supportsWorker: true,
  },
  {
    id: "HuggingFaceTB/SmolVLM-256M-Instruct",
    name: "SmolVLM 256M (Vision)",
    device: "webgpu",
    dtype: "fp32",
    isVisionModel: true,
    supportsWorker: true,
  },
];
