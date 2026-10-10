import {
  LanguageModelV4Prompt,
  LanguageModelV4ToolResultOutput,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import type {
  ChatCompletionMessage,
  ChatCompletionToolCall,
} from "@wllama/wllama/esm/index.js";

function toolResultToString(output: LanguageModelV4ToolResultOutput): string {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value;
    case "execution-denied":
      return JSON.stringify({
        error: "Tool execution was denied",
        reason: output.reason,
      });
    default:
      return JSON.stringify(output.value);
  }
}

/**
 * Convert an AI SDK prompt into wllama's OpenAI-compatible chat messages.
 *
 * Tool calls and tool results use the native OpenAI shapes
 * (`assistant.tool_calls` / `role: "tool"`), which llama.cpp's Jinja chat
 * templates understand. Only text input is supported.
 */
export function convertToWllamaMessages(
  prompt: LanguageModelV4Prompt,
): ChatCompletionMessage[] {
  const messages: ChatCompletionMessage[] = [];

  for (const message of prompt) {
    switch (message.role) {
      case "system":
        messages.push({ role: "system", content: message.content });
        break;

      case "user": {
        const texts: string[] = [];
        for (const part of message.content) {
          if (part.type !== "text") {
            throw new UnsupportedFunctionalityError({
              functionality: `${part.type} input`,
            });
          }
          texts.push(part.text);
        }
        messages.push({ role: "user", content: texts.join("\n") });
        break;
      }

      case "assistant": {
        let text = "";
        const toolCalls: ChatCompletionToolCall[] = [];
        for (const part of message.content) {
          if (part.type === "text") {
            text += part.text;
          } else if (part.type === "tool-call") {
            toolCalls.push({
              id: part.toolCallId,
              type: "function",
              function: {
                name: part.toolName,
                arguments:
                  typeof part.input === "string"
                    ? part.input
                    : JSON.stringify(part.input ?? {}),
              },
            });
          }
          // reasoning and other parts are not sent back to the model
        }
        if (text || toolCalls.length > 0) {
          messages.push({
            role: "assistant",
            content: text || null,
            ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
          });
        }
        break;
      }

      case "tool": {
        for (const part of message.content) {
          if (part.type !== "tool-result") continue;
          messages.push({
            role: "tool",
            tool_call_id: part.toolCallId,
            content: toolResultToString(part.output),
          });
        }
        break;
      }
    }
  }

  return messages;
}
