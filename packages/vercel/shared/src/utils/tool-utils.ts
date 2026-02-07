/**
 * Utilities for working with AI SDK tools
 */

import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3ProviderTool,
} from "@ai-sdk/provider";

/**
 * Type guard to check if a tool is a function tool
 *
 * @param tool - The tool to check
 * @returns true if the tool is a LanguageModelV3FunctionTool
 */
export function isFunctionTool(
  tool: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool,
): tool is LanguageModelV3FunctionTool {
  return tool.type === "function";
}
