import type {
  JSONSchema7,
  LanguageModelV3FunctionTool,
} from "@ai-sdk/provider";

/**
 * JSON Schema definition for tool parameters
 * Compatible with JSON Schema Draft 7
 */
export type JSONSchema = JSONSchema7;

/**
 * Tool definition in AI SDK format (function tools only)
 */
export type ToolDefinition = Pick<
  LanguageModelV3FunctionTool,
  "name" | "description"
> & {
  parameters: JSONSchema7;
};

/**
 * Parsed tool call from JSON response
 */
export interface ParsedToolCall {
  type: "tool-call";
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
}

/**
 * Tool execution result
 */
export interface ToolResult {
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError?: boolean;
}

/**
 * Result of parsing a response that may contain tool calls
 */
export interface ParsedResponse {
  toolCalls: ParsedToolCall[];
  textContent: string;
}
