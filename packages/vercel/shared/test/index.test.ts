import { describe, it, expect } from "vitest";
import {
  isFunctionTool,
  createUnsupportedSettingWarning,
  createUnsupportedToolWarning,
  buildJsonToolSystemPrompt,
  parseJsonFunctionCalls,
  formatToolResults,
  ToolCallFenceDetector,
} from "../src";

describe("@browser-ai/shared exports", () => {
  describe("isFunctionTool", () => {
    it("should return true for function tools", () => {
      const functionTool = {
        type: "function" as const,
        name: "test",
        inputSchema: { type: "object" as const },
      };
      expect(isFunctionTool(functionTool)).toBe(true);
    });

    it("should return false for provider tools", () => {
      const providerTool = {
        type: "provider",
        name: "test",
        id: "test.provider",
        args: {},
      } as const;
      expect(isFunctionTool(providerTool)).toBe(false);
    });
  });

  describe("createUnsupportedSettingWarning", () => {
    it("should create a warning object", () => {
      const warning = createUnsupportedSettingWarning(
        "maxTokens",
        "Not supported",
      );
      expect(warning).toEqual({
        type: "unsupported",
        feature: "maxTokens",
        details: "Not supported",
      });
    });
  });

  describe("createUnsupportedToolWarning", () => {
    it("should create a warning object with tool name", () => {
      const tool = {
        type: "provider",
        name: "customTool",
        id: "custom.tool",
        args: {},
      } as const;
      const warning = createUnsupportedToolWarning(tool, "Not supported");
      expect(warning).toEqual({
        type: "unsupported",
        feature: "tool:customTool",
        details: "Not supported",
      });
    });
  });

  describe("buildJsonToolSystemPrompt", () => {
    it("should return empty string for no tools", () => {
      const result = buildJsonToolSystemPrompt(undefined, []);
      expect(result).toBe("");
    });

    it("should return original prompt for no tools", () => {
      const result = buildJsonToolSystemPrompt("Hello", []);
      expect(result).toBe("Hello");
    });

    it("should build prompt with tools", () => {
      const result = buildJsonToolSystemPrompt(undefined, [
        {
          name: "test",
          description: "Test tool",
          parameters: { type: "object" },
        },
      ]);
      expect(result).toContain("Available Tools");
      expect(result).toContain("test");
    });
  });

  describe("parseJsonFunctionCalls", () => {
    it("should return empty for no tool calls", () => {
      const result = parseJsonFunctionCalls("Hello world");
      expect(result.toolCalls).toHaveLength(0);
      expect(result.textContent).toBe("Hello world");
    });

    it("should parse tool call fences", () => {
      const input = '```tool_call\n{"name": "test", "arguments": {}}\n```';
      const result = parseJsonFunctionCalls(input);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("test");
    });
  });

  describe("formatToolResults", () => {
    it("should return empty string for no results", () => {
      expect(formatToolResults([])).toBe("");
    });

    it("should format tool results", () => {
      const result = formatToolResults([
        { toolCallId: "1", toolName: "test", result: { data: "ok" } },
      ]);
      expect(result).toContain("tool_result");
      expect(result).toContain("test");
    });
  });

  describe("ToolCallFenceDetector", () => {
    it("should create instance", () => {
      const detector = new ToolCallFenceDetector();
      expect(detector).toBeInstanceOf(ToolCallFenceDetector);
    });

    it("should detect complete fence", () => {
      const detector = new ToolCallFenceDetector();
      detector.addChunk('```tool_call\n{"name": "test"}\n```');
      const result = detector.detectFence();
      expect(result.fence).not.toBeNull();
    });
  });
});
