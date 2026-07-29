/**
 * Utilities for working with AI SDK tools
 */

import type {
  LanguageModelV4FunctionTool,
  LanguageModelV4ProviderTool,
} from "@ai-sdk/provider";

/**
 * Type guard to check if a tool is a function tool
 *
 * @param tool - The tool to check
 * @returns true if the tool is a LanguageModelV4FunctionTool
 */
export function isFunctionTool(
  tool: LanguageModelV4FunctionTool | LanguageModelV4ProviderTool,
): tool is LanguageModelV4FunctionTool {
  return tool.type === "function";
}
