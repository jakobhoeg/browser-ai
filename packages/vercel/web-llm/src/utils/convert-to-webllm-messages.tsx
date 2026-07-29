import {
  LanguageModelV4Prompt,
  LanguageModelV4ToolResultPart,
  LanguageModelV4ToolResultOutput,
  SharedV4FileData,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import * as webllm from "@mlc-ai/web-llm";
import { formatToolResults, type ToolResult } from "@browser-ai/shared";

/**
 * Converts the AI SDK ToolResultOutput format to a simple value + error flag
 */
function convertToolResultOutput(output: LanguageModelV4ToolResultOutput): {
  value: unknown;
  isError: boolean;
} {
  switch (output.type) {
    case "text":
      return { value: output.value, isError: false };
    case "json":
      return { value: output.value, isError: false };
    case "error-text":
      return { value: output.value, isError: true };
    case "error-json":
      return { value: output.value, isError: true };
    case "content":
      return { value: output.value, isError: false };
    case "execution-denied":
      return { value: output.reason, isError: true };
    default: {
      const exhaustiveCheck: never = output;
      return { value: exhaustiveCheck, isError: false };
    }
  }
}

/**
 * Converts a ToolResultPart to our internal ToolResult format
 */
function toToolResult(part: LanguageModelV4ToolResultPart): ToolResult {
  const { value, isError } = convertToolResultOutput(part.output);
  return {
    toolCallId: part.toolCallId,
    toolName: part.toolName,
    result: value,
    isError,
  };
}

function uint8ArrayToBase64(uint8array: Uint8Array): string {
  const binary = Array.from(uint8array, (byte) =>
    String.fromCharCode(byte),
  ).join("");
  return btoa(binary);
}

function convertDataToURL(data: SharedV4FileData, mediaType: string): string {
  switch (data.type) {
    case "url":
      return data.url.toString();

    case "data":
      return data.data instanceof Uint8Array
        ? `data:${mediaType};base64,${uint8ArrayToBase64(data.data)}`
        : // AI SDK provides base64 string
          `data:${mediaType};base64,${data.data}`;

    default:
      throw new UnsupportedFunctionalityError({
        functionality: `file data type: ${data.type}`,
      });
  }
}

export function convertToWebLLMMessages(
  prompt: LanguageModelV4Prompt,
): webllm.ChatCompletionMessageParam[] {
  const messages: webllm.ChatCompletionMessageParam[] = [];

  for (const message of prompt) {
    switch (message.role) {
      case "system":
        messages.push({
          role: "system",
          content: message.content,
        });
        break;

      case "user":
        const hasFileContent = message.content.some(
          (part) => part.type === "file",
        );

        if (!hasFileContent) {
          const userContent: string[] = [];
          for (const part of message.content) {
            if (part.type === "text") {
              userContent.push(part.text);
            }
          }
          messages.push({
            role: "user",
            content: userContent.join("\n"),
          });
          break;
        }

        const content: webllm.ChatCompletionContentPart[] = [];
        for (const part of message.content) {
          if (part.type === "text") {
            content.push({ type: "text", text: part.text });
          } else if (part.type === "file") {
            if (!part.mediaType?.startsWith("image/")) {
              throw new UnsupportedFunctionalityError({
                functionality: `file input with media type '${part.mediaType}'`,
              });
            }
            content.push({
              type: "image_url",
              image_url: {
                url: convertDataToURL(part.data, part.mediaType),
              },
            });
          }
        }
        messages.push({ role: "user", content });
        break;

      case "assistant":
        let assistantContent = "";
        const toolCallsInMessage: Array<{
          toolCallId: string;
          toolName: string;
        }> = [];

        for (const part of message.content) {
          if (part.type === "text") {
            assistantContent += part.text;
          } else if (part.type === "tool-call") {
            // Store tool call info but don't include in content
            // Tool calls will be tracked separately
            toolCallsInMessage.push({
              toolCallId: part.toolCallId,
              toolName: part.toolName,
            });
          }
        }

        // Only add assistant message if there's text content
        // Tool calls are handled via the JSON fence format in the text
        if (assistantContent) {
          messages.push({
            role: "assistant",
            content: assistantContent,
          });
        }
        break;

      case "tool":
        // Collect tool results and format them
        // filter for tool-result parts only
        // not sure how to support tool-approval-response parts yet
        const toolResults: ToolResult[] = message.content
          .filter((part) => part.type === "tool-result")
          .map(toToolResult);

        // Format tool results as user message with JSON fence format
        const formattedResults = formatToolResults(toolResults);
        messages.push({
          role: "user",
          content: formattedResults,
        });
        break;
    }
  }

  return messages;
}
