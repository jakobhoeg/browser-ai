import { describe, it, expect } from "vitest";
import { convertToWebLLMMessages } from "../src/utils/convert-to-webllm-messages";
import {
  LanguageModelV3Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";

describe("convertToWebLLMMessages", () => {
  describe("text messages", () => {
    it("should convert simple text user message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, world!" }],
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result).toEqual([
        {
          role: "user",
          content: "Hello, world!",
        },
      ]);
    });

    it("should convert system message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "system",
          content: "You are a helpful assistant.",
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result).toEqual([
        {
          role: "system",
          content: "You are a helpful assistant.",
        },
      ]);
    });

    it("should convert assistant message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [{ type: "text", text: "Hi there!" }],
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result).toEqual([
        {
          role: "assistant",
          content: "Hi there!",
        },
      ]);
    });

    it("should handle conversation with multiple message types", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "system",
          content: "You are helpful.",
        },
        {
          role: "user",
          content: [{ type: "text", text: "Hello" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Hi!" }],
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result).toEqual([
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi!" },
      ]);
    });
  });

  describe("image file conversion", () => {
    it("should convert base64 image data to data URL", () => {
      const base64Data = "SGVsbG8gV29ybGQ="; // "Hello World" in base64
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            {
              type: "file",
              mediaType: "image/png",
              data: base64Data,
            },
          ],
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image_url",
            image_url: {
              url: `data:image/png;base64,${base64Data}`,
            },
          },
        ],
      });
    });

    it("should handle mixed text and image content", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "text", text: "Look at this image:" },
            {
              type: "file",
              mediaType: "image/png",
              data: "aGVsbG8=", // "hello" in base64
            },
            { type: "text", text: "What do you see?" },
          ],
        },
      ];

      const result = convertToWebLLMMessages(prompt);

      expect(result[0]).toEqual({
        role: "user",
        content: [
          { type: "text", text: "Look at this image:" },
          {
            type: "image_url",
            image_url: {
              url: "data:image/png;base64,aGVsbG8=",
            },
          },
          { type: "text", text: "What do you see?" },
        ],
      });
    });
  });

  describe("error handling", () => {
    it("should throw for non-image file types", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "audio/mp3",
              data: "some data",
            },
          ],
        },
      ];

      expect(() => convertToWebLLMMessages(prompt)).toThrow(
        UnsupportedFunctionalityError,
      );
      expect(() => convertToWebLLMMessages(prompt)).toThrow(
        "file input with media type 'audio/mp3'",
      );
    });
  });

  describe("edge cases", () => {
    it("should handle empty prompt", () => {
      const result = convertToWebLLMMessages([]);
      expect(result).toEqual([]);
    });
  });
});
