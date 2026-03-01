import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { BrowserAIEmbeddingModel } from "../src/embedding/browser-ai-embedding-model";
import { TextEmbedder } from "@mediapipe/tasks-text";

vi.mock("@mediapipe/tasks-text", () => ({
  TextEmbedder: {
    createFromOptions: vi.fn(),
  },
}));

describe("BrowserAIEmbeddingModel", () => {
  let mockTextEmbedder: any;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock global fetch for model asset loading
    const mockReader = {
      read: vi.fn().mockResolvedValue({ done: true }),
    };

    const mockResponse = {
      body: {
        getReader: vi.fn().mockReturnValue(mockReader),
      },
    };

    (global as any).fetch = vi.fn().mockResolvedValue(mockResponse);

    // Create mock embedder instance
    mockTextEmbedder = {
      embed: vi.fn(),
    };

    // Setup default mock implementations
    vi.mocked(TextEmbedder.createFromOptions).mockResolvedValue(
      mockTextEmbedder,
    );

    mockTextEmbedder.embed.mockReturnValue({
      embeddings: [
        {
          floatEmbedding: [0.1, 0.2, 0.3, 0.4, 0.5],
          headIndex: 0,
          headName: "embedding",
        },
      ],
    });
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.unstubAllGlobals();
  });

  describe("Construction", () => {
    it("should instantiate correctly with default settings", () => {
      const model = new BrowserAIEmbeddingModel();

      expect(model).toBeInstanceOf(BrowserAIEmbeddingModel);
      expect(model.modelId).toBe("embedding");
      expect(model.provider).toBe("google-mediapipe");
      expect(model.specificationVersion).toBe("v3");
      expect(model.supportsParallelCalls).toBe(true);
      expect(model.maxEmbeddingsPerCall).toBeUndefined();
    });

    it("should instantiate with custom settings", () => {
      const customSettings = {
        modelAssetPath: "https://custom-model-path.tflite",
        wasmLoaderPath: "https://custom-loader.js",
        wasmBinaryPath: "https://custom-binary.wasm",
        l2Normalize: true,
        quantize: false,
        delegate: "GPU" as const,
      };

      const model = new BrowserAIEmbeddingModel(customSettings);
      expect(model).toBeInstanceOf(BrowserAIEmbeddingModel);
    });
  });

  describe("doEmbed", () => {
    it("should generate embeddings for single text", async () => {
      const model = new BrowserAIEmbeddingModel();

      const result = await model.doEmbed({
        values: ["Hello, world!"],
      });

      expect(result.embeddings).toHaveLength(1);
      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
      expect(result.providerMetadata?.mediapipe).toEqual({
        model: "universal_sentence_encoder",
        provider: "google-mediapipe",
        processed_texts: 1,
      });
      expect(mockTextEmbedder.embed).toHaveBeenCalledWith("Hello, world!");
    });

    it("should generate embeddings for multiple texts", async () => {
      const model = new BrowserAIEmbeddingModel();

      // Mock different embeddings for different texts
      mockTextEmbedder.embed
        .mockReturnValueOnce({
          embeddings: [
            {
              floatEmbedding: [0.1, 0.2, 0.3],
              headIndex: 0,
              headName: "embedding",
            },
          ],
        })
        .mockReturnValueOnce({
          embeddings: [
            {
              floatEmbedding: [0.4, 0.5, 0.6],
              headIndex: 0,
              headName: "embedding",
            },
          ],
        });

      const result = await model.doEmbed({
        values: ["First text", "Second text"],
      });

      expect(result.embeddings).toHaveLength(2);
      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3]);
      expect(result.embeddings[1]).toEqual([0.4, 0.5, 0.6]);
      expect(result.providerMetadata?.mediapipe.processed_texts).toBe(2);
      expect(mockTextEmbedder.embed).toHaveBeenCalledTimes(2);
    });

    it("should handle empty embeddings gracefully", async () => {
      const model = new BrowserAIEmbeddingModel();

      mockTextEmbedder.embed.mockReturnValue({
        embeddings: [],
      });

      const result = await model.doEmbed({
        values: ["Test text"],
      });

      expect(result.embeddings).toHaveLength(1);
      expect(result.embeddings[0]).toEqual([]);
    });

    it("should handle empty values array", async () => {
      const model = new BrowserAIEmbeddingModel();

      const result = await model.doEmbed({
        values: [],
      });

      expect(result.embeddings).toHaveLength(0);
      expect(result.providerMetadata?.mediapipe.processed_texts).toBe(0);
      expect(mockTextEmbedder.embed).not.toHaveBeenCalled();
    });
  });

  describe("Abort Signal Handling", () => {
    it("should throw error when signal is already aborted", async () => {
      const model = new BrowserAIEmbeddingModel();
      const abortController = new AbortController();
      abortController.abort();

      await expect(
        model.doEmbed({
          values: ["Test text"],
          abortSignal: abortController.signal,
        }),
      ).rejects.toThrow("Operation was aborted");
    });

    it("should work without abort signal", async () => {
      const model = new BrowserAIEmbeddingModel();

      const result = await model.doEmbed({
        values: ["Test text"],
      });

      expect(result.embeddings).toHaveLength(1);
      expect(result.embeddings[0]).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
    });
  });

  describe("Error Handling", () => {
    it("should handle TextEmbedder creation errors", async () => {
      vi.mocked(TextEmbedder.createFromOptions).mockRejectedValue(
        new Error("Failed to create embedder"),
      );

      const model = new BrowserAIEmbeddingModel();

      await expect(model.doEmbed({ values: ["test"] })).rejects.toThrow(
        "Failed to create embedder",
      );
    });

    it("should handle embedding errors", async () => {
      mockTextEmbedder.embed.mockImplementation(() => {
        throw new Error("Embedding failed");
      });

      const model = new BrowserAIEmbeddingModel();

      await expect(model.doEmbed({ values: ["test"] })).rejects.toThrow(
        "Embedding failed",
      );
    });
  });

  describe("Integration Tests", () => {
    it("should maintain embedder instance across multiple calls", async () => {
      const model = new BrowserAIEmbeddingModel();

      await model.doEmbed({ values: ["First call"] });
      await model.doEmbed({ values: ["Second call"] });

      // Should only create embedder once
      expect(vi.mocked(TextEmbedder.createFromOptions)).toHaveBeenCalledTimes(
        1,
      );
      expect(mockTextEmbedder.embed).toHaveBeenCalledTimes(2);
    });
  });
});
