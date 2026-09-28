import { describe, it, expect } from "vitest";
import { convertToLiteRTMessages } from "../src/utils/convert-to-litert-messages";
import {
  LanguageModelV4Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";

describe("convertToLiteRTMessages", () => {
  describe("text messages", () => {
    it("should convert a simple text user message", () => {
      const prompt: LanguageModelV4Prompt = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, world!" }],
        },
      ];

      expect(convertToLiteRTMessages(prompt)).toEqual([
        { role: "user", content: "Hello, world!" },
      ]);
    });

    it("should convert a system message", () => {
      const prompt: LanguageModelV4Prompt = [
        { role: "system", content: "You are a helpful assistant." },
      ];

      expect(convertToLiteRTMessages(prompt)).toEqual([
        { role: "system", content: "You are a helpful assistant." },
      ]);
    });

    it("should convert an assistant message", () => {
      const prompt: LanguageModelV4Prompt = [
        {
          role: "assistant",
          content: [{ type: "text", text: "Hi there!" }],
        },
      ];

      expect(convertToLiteRTMessages(prompt)).toEqual([
        { role: "assistant", content: "Hi there!" },
      ]);
    });

    it("should handle a multi-turn conversation", () => {
      const prompt: LanguageModelV4Prompt = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: [{ type: "text", text: "Hello" }] },
        { role: "assistant", content: [{ type: "text", text: "Hi!" }] },
      ];

      expect(convertToLiteRTMessages(prompt)).toEqual([
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
      ]);
    });
  });

  describe("error handling", () => {
    it("should throw for file inputs (text-only web runtime)", () => {
      const prompt: LanguageModelV4Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/png",
              data: { type: "data", data: "aGVsbG8=" },
            },
          ],
        },
      ];

      expect(() => convertToLiteRTMessages(prompt)).toThrow(
        UnsupportedFunctionalityError,
      );
    });
  });

  describe("edge cases", () => {
    it("should handle an empty prompt", () => {
      expect(convertToLiteRTMessages([])).toEqual([]);
    });
  });
});
