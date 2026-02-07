import type {
  JSONSchema7,
  LanguageModelV3FunctionTool,
} from "@ai-sdk/provider";
import type { ToolDefinition } from "@browser-ai/shared";

/**
 * HuggingFace tool definition format compatible with apply_chat_template
 */
export interface HuggingFaceToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: JSONSchema7;
  };
}

/**
 * Converts AI SDK tool format to HuggingFace chat template tool format.
 * The converted tools can be passed to processor.apply_chat_template({ tools: [...] })
 * to enable native tool-calling support in models that have it in their chat templates.
 *
 * @param tools - Array of AI SDK function tools or ToolDefinition
 * @returns Array of HuggingFace-formatted tool definitions
 */
export function convertToolsToHuggingFaceFormat(
  tools: Array<LanguageModelV3FunctionTool | ToolDefinition>,
): HuggingFaceToolDefinition[] {
  return tools.map((tool) => {
    const parameters =
      "inputSchema" in tool ? tool.inputSchema : tool.parameters;

    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description ?? "",
        parameters,
      },
    };
  });
}
