import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";
import { WebLLMEmbeddingModel, webLLM } from "../src";

const mockEmbeddingsCreate = vi.fn();
const mockReload = vi.fn();
const mockEngineConstructor = vi.fn();
const mockInterruptGenerate = vi.fn();

vi.mock("@mlc-ai/web-llm", () => ({
  MLCEngine: vi.fn().mockImplementation(function (config) {
    mockEngineConstructor(config);
    return {
      embeddings: { create: mockEmbeddingsCreate },
      reload: mockReload,
      interruptGenerate: mockInterruptGenerate,
    };
  }),
  CreateWebWorkerMLCEngine: vi.fn().mockImplementation(() =>
    Promise.resolve({
      embeddings: { create: mockEmbeddingsCreate },
      interruptGenerate: mockInterruptGenerate,
    }),
  ),
}));

describe("WebLLMEmbeddingModel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(global.navigator, "gpu", {
      value: {},
      configurable: true,
    });

    mockEmbeddingsCreate.mockResolvedValue({
      data: [{ index: 0, embedding: [0.1, 0.2, 0.3], object: "embedding" }],
      model: "test-model",
      usage: {
        prompt_tokens: 5,
        total_tokens: 5,
        extra: { prefill_tokens_per_s: 100 },
      },
    });
  });

  afterEach(() => {
    Object.defineProperty(global.navigator, "gpu", {
      value: undefined,
      configurable: true,
    });
  });

  describe("Construction", () => {
    it("should instantiate correctly with default settings", () => {
      const model = new WebLLMEmbeddingModel("test-model");

      expect(model).toBeInstanceOf(WebLLMEmbeddingModel);
      expect(model.modelId).toBe("test-model");
      expect(model.provider).toBe("web-llm");
      expect(model.specificationVersion).toBe("v3");
      expect(model.supportsParallelCalls).toBe(false);
      expect(model.maxEmbeddingsPerCall).toBe(100);
    });

    it("should instantiate with custom settings", () => {
      const model = new WebLLMEmbeddingModel("test-model", {
        maxEmbeddingsPerCall: 25,
      });

      expect(model.maxEmbeddingsPerCall).toBe(25);
    });

    it("should work with factory function", () => {
      const model = webLLM.embeddingModel("test-model", {
        maxEmbeddingsPerCall: 50,
      });

      expect(model).toBeInstanceOf(WebLLMEmbeddingModel);
      expect(model.maxEmbeddingsPerCall).toBe(50);
    });
  });

  describe("Availability", () => {
    it("should return 'unavailable' without WebGPU", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });

      const model = new WebLLMEmbeddingModel("test-model");
      expect(await model.availability()).toBe("unavailable");
    });

    it("should return 'downloadable' with WebGPU", async () => {
      const model = new WebLLMEmbeddingModel("test-model");
      expect(await model.availability()).toBe("downloadable");
    });

    it("should return 'available' after initialization", async () => {
      const model = new WebLLMEmbeddingModel("test-model");
      (model as any).isInitialized = true;
      expect(await model.availability()).toBe("available");
    });
  });

  describe("doEmbed", () => {
    it("should generate embeddings for single text", async () => {
      const model = new WebLLMEmbeddingModel("test-model");

      const result = await model.doEmbed({ values: ["hello"] });

      expect(result.embeddings).toHaveLength(1);
      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3]);
      expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
        input: ["hello"],
        model: "test-model",
      });
    });

    it("should generate embeddings for multiple texts", async () => {
      mockEmbeddingsCreate.mockResolvedValue({
        data: [
          { index: 0, embedding: [0.1, 0.2, 0.3], object: "embedding" },
          { index: 1, embedding: [0.4, 0.5, 0.6], object: "embedding" },
        ],
        model: "test-model",
        usage: { prompt_tokens: 10, total_tokens: 10, extra: {} },
      });

      const model = new WebLLMEmbeddingModel("test-model");
      const result = await model.doEmbed({ values: ["hello", "world"] });

      expect(result.embeddings).toHaveLength(2);
      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3]);
      expect(result.embeddings[1]).toEqual([0.4, 0.5, 0.6]);
    });

    it("should sort embeddings by index", async () => {
      mockEmbeddingsCreate.mockResolvedValue({
        data: [
          { index: 1, embedding: [0.4, 0.5, 0.6], object: "embedding" },
          { index: 0, embedding: [0.1, 0.2, 0.3], object: "embedding" },
        ],
        model: "test-model",
        usage: { prompt_tokens: 10, total_tokens: 10, extra: {} },
      });

      const model = new WebLLMEmbeddingModel("test-model");
      const result = await model.doEmbed({ values: ["hello", "world"] });

      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3]);
      expect(result.embeddings[1]).toEqual([0.4, 0.5, 0.6]);
    });

    it("should include provider metadata", async () => {
      const model = new WebLLMEmbeddingModel("test-model");
      const result = await model.doEmbed({ values: ["hello"] });

      expect(result.providerMetadata).toEqual({
        webllm: {
          model: "test-model",
          promptTokens: 5,
          totalTokens: 5,
          prefillTokensPerSecond: 100,
        },
      });
    });
  });

  describe("Abort Signal Handling", () => {
    it("should throw error when signal is already aborted", async () => {
      const model = new WebLLMEmbeddingModel("test-model");
      const abortController = new AbortController();
      abortController.abort();

      await expect(
        model.doEmbed({
          values: ["hello"],
          abortSignal: abortController.signal,
        }),
      ).rejects.toThrow("Operation was aborted");
    });

    it("should call interruptGenerate when abort signal fires during embedding", async () => {
      const abortController = new AbortController();
      mockEmbeddingsCreate.mockImplementation(() => {
        abortController.abort();
        return Promise.resolve({
          data: [{ index: 0, embedding: [0.1, 0.2, 0.3], object: "embedding" }],
          model: "test-model",
          usage: { prompt_tokens: 5, total_tokens: 5, extra: {} },
        });
      });

      const model = new WebLLMEmbeddingModel("test-model");
      await model.doEmbed({
        values: ["hello"],
        abortSignal: abortController.signal,
      });

      expect(mockInterruptGenerate).toHaveBeenCalled();
    });
  });

  describe("Error Handling", () => {
    it("should throw LoadSettingError without WebGPU", async () => {
      Object.defineProperty(global.navigator, "gpu", {
        value: undefined,
        configurable: true,
      });

      const model = new WebLLMEmbeddingModel("test-model");

      await expect(model.doEmbed({ values: ["hello"] })).rejects.toThrow(
        "WebLLM is not available",
      );
    });

    it("should throw error when too many values provided", async () => {
      const model = new WebLLMEmbeddingModel("test-model", {
        maxEmbeddingsPerCall: 2,
      });

      await expect(model.doEmbed({ values: ["a", "b", "c"] })).rejects.toThrow(
        "Too many values",
      );
    });

    it("should handle embedding API errors", async () => {
      mockEmbeddingsCreate.mockRejectedValue(new Error("API Error"));

      const model = new WebLLMEmbeddingModel("test-model");

      await expect(model.doEmbed({ values: ["hello"] })).rejects.toThrow(
        "WebLLM embedding failed: API Error",
      );
    });
  });

  describe("Worker Engine", () => {
    it("should use CreateWebWorkerMLCEngine when worker is provided", async () => {
      const mockWorker = {} as Worker;
      const model = new WebLLMEmbeddingModel("test-model", {
        worker: mockWorker,
      });

      await model.doEmbed({ values: ["hello"] });

      expect(CreateWebWorkerMLCEngine).toHaveBeenCalledWith(
        mockWorker,
        "test-model",
        expect.any(Object),
      );
      expect(mockReload).not.toHaveBeenCalled();
    });
  });

  describe("Integration Tests", () => {
    it("should maintain engine instance across multiple calls", async () => {
      const model = new WebLLMEmbeddingModel("test-model");

      await model.doEmbed({ values: ["first"] });
      await model.doEmbed({ values: ["second"] });

      expect(mockReload).toHaveBeenCalledTimes(1);
      expect(mockEmbeddingsCreate).toHaveBeenCalledTimes(2);
    });

    it("should track initialization state", async () => {
      const model = new WebLLMEmbeddingModel("test-model");

      expect(model.isModelInitialized).toBe(false);
      await model.doEmbed({ values: ["test"] });
      expect(model.isModelInitialized).toBe(true);
    });

    it("should call progress callback during initialization", async () => {
      mockEngineConstructor.mockImplementation((config) => {
        config.initProgressCallback?.({ progress: 0.5, text: "loading" });
      });

      const model = new WebLLMEmbeddingModel("test-model");
      const progressCallback = vi.fn();

      await model.createSessionWithProgress(progressCallback);

      expect(progressCallback).toHaveBeenCalledWith({
        progress: 0.5,
        text: "loading",
      });
    });
  });
});
