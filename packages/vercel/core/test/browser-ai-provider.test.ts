import { describe, it, expect, vi, beforeEach } from "vitest";
import { browserAI, createBrowserAI } from "../src/browser-ai-provider";
import { BrowserAIChatLanguageModel } from "../src/chat/browser-ai-language-model";
import { BrowserAIEmbeddingModel } from "../src/embedding/browser-ai-embedding-model";

// Mock the dependencies
vi.mock("../src/chat/browser-ai-language-model", () => ({
  BrowserAIChatLanguageModel: vi.fn(),
}));

vi.mock("../src/embedding/browser-ai-embedding-model", () => ({
  BrowserAIEmbeddingModel: vi.fn(),
}));

describe("BrowserAI Provider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("createBrowserAI", () => {
    it("should create a provider with all expected methods", () => {
      const provider = createBrowserAI();

      expect(provider).toBeInstanceOf(Function);
      expect(provider.languageModel).toBeInstanceOf(Function);
      expect(provider.chat).toBeInstanceOf(Function);
      expect(provider.embedding).toBeInstanceOf(Function);
      expect(provider.embeddingModel).toBeInstanceOf(Function);
      expect(provider.imageModel).toBeInstanceOf(Function);
      expect(provider.speechModel).toBeInstanceOf(Function);
      expect(provider.transcriptionModel).toBeInstanceOf(Function);
    });

    it("should prevent calling with new keyword", () => {
      const provider = createBrowserAI();

      expect(() => {
        // @ts-expect-error - intentionally testing invalid usage
        new provider("text");
      }).toThrow(
        "The BrowserAI model function cannot be called with the new keyword.",
      );
    });
  });

  describe("Language Model Creation", () => {
    it("should create language model via direct call with explicit model ID", () => {
      const provider = createBrowserAI();
      provider("text", { temperature: 0.5 });

      expect(BrowserAIChatLanguageModel).toHaveBeenCalledWith("text", {
        temperature: 0.5,
      });
    });

    it("should create language model via direct call with default model ID", () => {
      const provider = createBrowserAI();
      provider(undefined, { temperature: 0.5 });

      expect(BrowserAIChatLanguageModel).toHaveBeenCalledWith("text", {
        temperature: 0.5,
      });
    });

    it("should create language model via direct call with no parameters", () => {
      const provider = createBrowserAI();
      provider();

      expect(BrowserAIChatLanguageModel).toHaveBeenCalledWith(
        "text",
        undefined,
      );
    });

    it("should create language model via languageModel method", () => {
      const provider = createBrowserAI();
      provider.languageModel("text", { temperature: 0.7 });

      expect(BrowserAIChatLanguageModel).toHaveBeenCalledWith("text", {
        temperature: 0.7,
      });
    });

    it("should create language model via chat method", () => {
      const provider = createBrowserAI();
      provider.chat("text", { temperature: 0.9 });

      expect(BrowserAIChatLanguageModel).toHaveBeenCalledWith("text", {
        temperature: 0.9,
      });
    });
  });

  describe("Embedding Model Creation", () => {
    it("should create embedding model via embedding method", () => {
      const provider = createBrowserAI();
      const settings = { l2Normalize: true };
      provider.embedding("embedding", settings);

      expect(BrowserAIEmbeddingModel).toHaveBeenCalledWith(settings);
    });

    it("should create embedding model via embeddingModel method", () => {
      const provider = createBrowserAI();
      const settings = { quantize: true };
      provider.embeddingModel("embedding", settings);

      expect(BrowserAIEmbeddingModel).toHaveBeenCalledWith(settings);
    });
  });

  describe("Unsupported Model Types", () => {
    it("should throw NoSuchModelError for image models", () => {
      const provider = createBrowserAI();

      expect(() => provider.imageModel("image")).toThrow();
    });

    it("should throw NoSuchModelError for speech models", () => {
      const provider = createBrowserAI();

      expect(() => provider.speechModel("speech")).toThrow();
    });

    it("should throw NoSuchModelError for transcription models", () => {
      const provider = createBrowserAI();

      expect(() => provider.transcriptionModel("transcribe")).toThrow();
    });
  });

  describe("Default Provider Instance", () => {
    it("should export a default provider instance", () => {
      expect(browserAI).toBeInstanceOf(Function);
      expect(browserAI.embedding).toBeInstanceOf(Function);
      expect(browserAI.chat).toBeInstanceOf(Function);
    });

    it("should work with the new API pattern", () => {
      browserAI.embedding("embedding", { l2Normalize: true });

      expect(BrowserAIEmbeddingModel).toHaveBeenCalledWith({
        l2Normalize: true,
      });
    });
  });
});
