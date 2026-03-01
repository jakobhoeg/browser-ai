import { describe, it, expect } from "vitest";
import { convertToBrowserAIMessages } from "../src/utils/convert-to-browser-ai-messages";
import {
  LanguageModelV3Prompt,
  LanguageModelV3Message,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";

describe("convertToBrowserAIMessages", () => {
  describe("text messages", () => {
    it("should convert simple text user message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, world!" }],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.systemMessage).toBeUndefined();
      expect(result.messages).toEqual([
        {
          role: "user",
          content: [{ type: "text", value: "Hello, world!" }],
        },
      ]);
    });

    it("should extract system message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "system",
          content: "You are a helpful assistant.",
        },
        {
          role: "user",
          content: [{ type: "text", text: "Hello!" }],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.systemMessage).toBe("You are a helpful assistant.");
      expect(result.messages).toEqual([
        {
          role: "user",
          content: [{ type: "text", value: "Hello!" }],
        },
      ]);
    });

    it("should convert assistant messages", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello!" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Hi there!" }],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages).toEqual([
        {
          role: "user",
          content: [{ type: "text", value: "Hello!" }],
        },
        {
          role: "assistant",
          content: "Hi there!",
        },
      ]);
    });
  });

  describe("image file conversion", () => {
    it("should convert base64 image data to Uint8Array", () => {
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

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages).toHaveLength(1);
      expect(result.messages[0].content).toHaveLength(2);
      expect(result.messages[0].content[0]).toEqual({
        type: "text",
        value: "What's in this image?",
      });

      const imageContent = result.messages[0].content[1] as any;
      expect(imageContent.type).toBe("image");
      expect(imageContent.value).toBeInstanceOf(Uint8Array);

      // Verify the conversion worked correctly
      const expectedBytes = new Uint8Array([
        72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100,
      ]);
      expect(imageContent.value).toEqual(expectedBytes);
    });

    it("should handle Uint8Array image data directly", () => {
      const uint8Data = new Uint8Array([1, 2, 3, 4]);
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/jpeg",
              data: uint8Data,
            },
          ],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content[0]).toEqual({
        type: "image",
        value: uint8Data,
      });
    });

    it("should handle URL image data", () => {
      const imageUrl = new URL("https://example.com/image.png");
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/webp",
              data: imageUrl,
            },
          ],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content[0]).toEqual({
        type: "image",
        value: "https://example.com/image.png",
      });
    });
  });

  describe("audio file conversion", () => {
    it("should convert base64 audio data to Uint8Array", () => {
      const base64Data = "UklGRnQAAABXQVZF"; // Valid WAV header start in base64
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this audio?" },
            {
              type: "file",
              mediaType: "audio/wav",
              data: base64Data,
            },
          ],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content[1]).toEqual({
        type: "audio",
        value: expect.any(Uint8Array),
      });
    });

    it("should handle Uint8Array audio data directly", () => {
      const audioData = new Uint8Array([82, 73, 70, 70]); // "RIFF" header
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "audio/mp3",
              data: audioData,
            },
          ],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content[0]).toEqual({
        type: "audio",
        value: audioData,
      });
    });
  });

  describe("mixed content", () => {
    it("should handle mixed text, image, and audio content", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            { type: "text", text: "Analyze this:" },
            {
              type: "file",
              mediaType: "image/png",
              data: "SGVsbG8=", // "Hello" in base64
            },
            { type: "text", text: "And this audio:" },
            {
              type: "file",
              mediaType: "audio/wav",
              data: new Uint8Array([1, 2, 3]),
            },
          ],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content).toHaveLength(4);
      expect(result.messages[0].content[0]).toEqual({
        type: "text",
        value: "Analyze this:",
      });
      expect(result.messages[0].content[1]).toEqual({
        type: "image",
        value: expect.any(Uint8Array),
      });
      expect(result.messages[0].content[2]).toEqual({
        type: "text",
        value: "And this audio:",
      });
      expect(result.messages[0].content[3]).toEqual({
        type: "audio",
        value: new Uint8Array([1, 2, 3]),
      });
    });
  });

  describe("tool support", () => {
    it("should convert assistant tool calls into toolCall fences", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check that." },
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "getWeather",
              input: { location: "Seattle" },
            },
          ],
        } as LanguageModelV3Message,
        {
          role: "user",
          content: [{ type: "text", text: "Thanks!" }],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0]).toEqual({
        role: "assistant",
        content:
          'Let me check that.\n```tool_call\n{"name":"getWeather","arguments":{"location":"Seattle"},"id":"call_123"}\n```',
      });
      expect(result.messages[1]).toEqual({
        role: "user",
        content: [{ type: "text", value: "Thanks!" }],
      });
    });

    it("should serialize scalar tool inputs as JSON strings", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_456",
              toolName: "getTemperature",
              input: "Zurich",
            },
          ],
        } as LanguageModelV3Message,
        {
          role: "user",
          content: [{ type: "text", text: "Appreciate it." }],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0]).toEqual({
        role: "assistant",
        content:
          '```tool_call\n{"name":"getTemperature","arguments":"Zurich","id":"call_456"}\n```',
      });
      expect(result.messages[1]).toEqual({
        role: "user",
        content: [{ type: "text", value: "Appreciate it." }],
      });
    });

    it("should handle parameter names with special characters", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_789",
              toolName: "register",
              input: { "123-name": "Alice" },
            },
          ],
        } as LanguageModelV3Message,
        {
          role: "user",
          content: [{ type: "text", text: "Continue." }],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content).toContain('"id":"call_789"');
      expect(result.messages[0].content).toContain('"123-name":"Alice"');
    });

    it("should deserialize JSON-stringified tool inputs", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_321",
              toolName: "search",
              input: JSON.stringify({ query: "status:open", limit: 5 }),
            },
          ],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages[0].content).toContain('"id":"call_321"');
      expect(result.messages[0].content).toContain(
        '"arguments":{"query":"status:open","limit":5}',
      );
    });

    it("should serialize multiple tool calls within a single fence", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_a",
              toolName: "first",
              input: { value: 1 },
            },
            {
              type: "tool-call",
              toolCallId: "call_b",
              toolName: "second",
              input: { value: 2 },
            },
          ],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      const content = result.messages[0].content as string;
      expect(content.match(/```tool_call/g)).toHaveLength(1);
      expect(content).toContain('"id":"call_a"');
      expect(content).toContain('"id":"call_b"');
      expect(content).toContain('"name":"first"');
      expect(content).toContain('"name":"second"');
    });

    it("should retain a trailing assistant tool call message", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check that." },
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "getWeather",
              input: { location: "Seattle" },
            },
          ],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages).toHaveLength(1);
      const content = result.messages[0].content as string;
      expect(content).toContain("Let me check that.");
      expect(content).toContain("```tool_call");
      expect(content).toContain('"id":"call_123"');
      expect(content).toContain('"name":"getWeather"');
    });

    it("should convert tool results into user tool_result blocks", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_123",
              toolName: "getWeather",
              output: {
                type: "json",
                value: { temperature: 70 },
              },
            },
          ],
        } as LanguageModelV3Message,
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages).toHaveLength(1);
      expect(result.messages[0]).toMatchObject({ role: "user" });
      const content = result.messages[0].content as string;
      expect(content).toContain("```tool_result");
      expect(content).toContain('"name":"getWeather"');
      expect(content).toContain('"id":"call_123"');
      expect(content).toContain('"result":{"temperature":70}');
    });
  });

  describe("error handling", () => {
    it("should throw for unsupported file types", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "video/mp4",
              data: "some data",
            },
          ],
        },
      ];

      expect(() => convertToBrowserAIMessages(prompt)).toThrow(
        UnsupportedFunctionalityError,
      );
    });

    it("should throw for unsupported content types", () => {
      const prompt = [
        {
          role: "user",
          content: [
            {
              type: "unsupported" as any,
              data: "some data",
            },
          ],
        },
      ] as LanguageModelV3Prompt;

      expect(() => convertToBrowserAIMessages(prompt)).toThrow(
        UnsupportedFunctionalityError,
      );
    });

    it("should throw for invalid base64 data", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/png",
              data: "invalid-base64-data!@#",
            },
          ],
        },
      ];

      expect(() => convertToBrowserAIMessages(prompt)).toThrow();
    });
  });

  describe("edge cases", () => {
    it("should handle empty prompt", () => {
      const result = convertToBrowserAIMessages([]);

      expect(result.systemMessage).toBeUndefined();
      expect(result.messages).toEqual([]);
    });

    it("should handle empty content arrays", () => {
      const prompt: LanguageModelV3Prompt = [
        {
          role: "user",
          content: [],
        },
      ];

      const result = convertToBrowserAIMessages(prompt);

      expect(result.messages).toEqual([
        {
          role: "user",
          content: [],
        },
      ]);
    });
  });
});
