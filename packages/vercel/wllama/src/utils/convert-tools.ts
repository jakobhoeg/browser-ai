import type {
  LanguageModelV4CallOptions,
  LanguageModelV4FunctionTool,
} from "@ai-sdk/provider";
import type {
  ChatCompletionParams,
  ChatCompletionTool,
  ChatCompletionToolChoice,
  ChatCompletionToolFunctionParameters,
} from "@wllama/wllama/esm/index.js";

/**
 * Convert AI SDK function tools to wllama's OpenAI-compatible tool definitions.
 */
export function convertToWllamaTools(
  tools: LanguageModelV4FunctionTool[],
): ChatCompletionTool[] {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      // wllama's type is a narrow subset of JSON Schema 7 object schemas
      parameters:
        tool.inputSchema as unknown as ChatCompletionToolFunctionParameters,
    },
  }));
}

/**
 * Convert the AI SDK tool choice to wllama's OpenAI-compatible tool choice.
 */
export function convertToWllamaToolChoice(
  toolChoice: LanguageModelV4CallOptions["toolChoice"],
): ChatCompletionToolChoice | undefined {
  switch (toolChoice?.type) {
    case "auto":
    case "none":
    case "required":
      return toolChoice.type;
    case "tool":
      return { type: "function", function: { name: toolChoice.toolName } };
    default:
      return undefined;
  }
}

/**
 * Convert the AI SDK response format to wllama's `response_format`.
 * JSON schemas are enforced by llama.cpp's grammar-constrained sampling.
 */
export function convertToWllamaResponseFormat(
  responseFormat: LanguageModelV4CallOptions["responseFormat"],
): ChatCompletionParams["response_format"] {
  if (responseFormat?.type !== "json") return undefined;
  if (!responseFormat.schema) return { type: "json_object" };
  return {
    type: "json_schema",
    json_schema: {
      name: responseFormat.name ?? "response",
      schema: responseFormat.schema,
    },
  };
}
