import { describe, it, expect } from "vitest";
import { parseModelId } from "../src/utils/model-source";

describe("parseModelId", () => {
  it("parses direct URLs", () => {
    expect(
      parseModelId("https://example.com/a/model-00001-of-00003.gguf"),
    ).toEqual({
      type: "url",
      url: "https://example.com/a/model-00001-of-00003.gguf",
    });
  });

  it("parses hf: repo + file", () => {
    expect(
      parseModelId("hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"),
    ).toEqual({
      type: "hf",
      params: {
        repo: "ggml-org/Qwen3-0.6B-GGUF",
        file: "Qwen3-0.6B-Q8_0.gguf",
      },
    });
  });

  it("parses nested file paths", () => {
    expect(
      parseModelId("hf:ggml-org/models/tinyllamas/stories260K.gguf"),
    ).toEqual({
      type: "hf",
      params: { repo: "ggml-org/models", file: "tinyllamas/stories260K.gguf" },
    });
  });

  it("rejects invalid ids", () => {
    expect(() => parseModelId("")).toThrow();
    expect(() => parseModelId("just-a-name")).toThrow();
    expect(() => parseModelId("hf:owner/repo")).toThrow();
    expect(() => parseModelId("owner/repo/file.gguf")).toThrow();
  });
});
