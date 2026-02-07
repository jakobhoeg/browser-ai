import {
  LanguageModelV3Prompt,
  LanguageModelV3ToolCallPart,
  LanguageModelV3ToolResultPart,
  LanguageModelV3ToolResultOutput,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { formatToolResults, type ToolResult } from "@browser-ai/shared";
export interface ConvertedMessages {
  systemMessage?: string;
  messages: LanguageModelMessage[];
}

/**
 * Convert base64 string to Uint8Array for browser AI compatibility
 * Browser AI supports BufferSource (including Uint8Array) for image/audio data
 */
function convertBase64ToUint8Array(base64: string): Uint8Array {
  try {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  } catch (error) {
    throw new Error(`Failed to convert base64 to Uint8Array: ${error}`);
  }
}

/**
 * Convert file data to the appropriate format for browser AI
 * Browser AI supports: Blob, BufferSource (Uint8Array), URLs
 */
function convertFileData(
  data: URL | Uint8Array | string,
  mediaType: string,
): Uint8Array | string {
  // Handle different data types from Vercel AI SDK
  if (data instanceof URL) {
    // URLs - keep as string (if supported by provider)
    return data.toString();
  }

  if (data instanceof Uint8Array) {
    // Already in correct format
    return data;
  }

  if (typeof data === "string") {
    // Base64 string from AI SDK - convert to Uint8Array
    return convertBase64ToUint8Array(data);
  }

  // Exhaustive check - this should never happen with the union type
  const exhaustiveCheck: never = data;
  throw new Error(`Unexpected data type for ${mediaType}: ${exhaustiveCheck}`);
}

function normalizeToolArguments(input: unknown): unknown {
  if (input === undefined) {
    return {};
  }

  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return input;
    }
  }

  return input ?? {};
}

function formatToolCallsJson(parts: LanguageModelV3ToolCallPart[]): string {
  if (!parts.length) {
    return "";
  }

  const payloads = parts.map((call) => {
    const payload: Record<string, unknown> = {
      name: call.toolName,
      arguments: normalizeToolArguments(call.input),
    };

    if (call.toolCallId) {
      payload.id = call.toolCallId;
    }

    return JSON.stringify(payload);
  });

  return `\`\`\`tool_call
${payloads.join("\n")}
\`\`\``;
}

function convertToolResultOutput(output: LanguageModelV3ToolResultOutput): {
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

function toToolResult(part: LanguageModelV3ToolResultPart): ToolResult {
  const { value, isError } = convertToolResultOutput(part.output);
  return {
    toolCallId: part.toolCallId,
    toolName: part.toolName,
    result: value,
    isError,
  };
}

/**
 * Convert Vercel AI SDK prompt format to browser AI Prompt API format
 * Returns system message (for initialPrompts) and regular messages (for prompt method)
 */
export function convertToBrowserAIMessages(
  prompt: LanguageModelV3Prompt,
): ConvertedMessages {
  const normalizedPrompt = prompt.slice();

  let systemMessage: string | undefined;
  const messages: LanguageModelMessage[] = [];

  for (const message of normalizedPrompt) {
    switch (message.role) {
      case "system": {
        // There's only ever one system message from AI SDK
        systemMessage = message.content;
        break;
      }

      case "user": {
        messages.push({
          role: "user",
          content: message.content.map((part) => {
            switch (part.type) {
              case "text": {
                return {
                  type: "text",
                  value: part.text,
                } as LanguageModelMessageContent;
              }

              case "file": {
                const { mediaType, data } = part;

                if (mediaType?.startsWith("image/")) {
                  const convertedData = convertFileData(data, mediaType);

                  return {
                    type: "image",
                    value: convertedData,
                  } as LanguageModelMessageContent;
                } else if (mediaType?.startsWith("audio/")) {
                  const convertedData = convertFileData(data, mediaType);

                  return {
                    type: "audio",
                    value: convertedData,
                  } as LanguageModelMessageContent;
                } else {
                  throw new UnsupportedFunctionalityError({
                    functionality: `file type: ${mediaType}`,
                  });
                }
              }

              default: {
                const exhaustiveCheck: never = part;
                throw new UnsupportedFunctionalityError({
                  functionality: `content type: ${(exhaustiveCheck as { type?: string }).type ?? "unknown"}`,
                });
              }
            }
          }),
        } as LanguageModelMessage);
        break;
      }

      case "assistant": {
        let text = "";
        const toolCallParts: LanguageModelV3ToolCallPart[] = [];

        for (const part of message.content) {
          switch (part.type) {
            case "text": {
              text += part.text;
              break;
            }
            case "reasoning": {
              text += part.text;
              break;
            }
            case "tool-call": {
              toolCallParts.push(part);
              break;
            }
            case "file": {
              throw new UnsupportedFunctionalityError({
                functionality: "assistant file attachments",
              });
            }
            case "tool-result": {
              throw new UnsupportedFunctionalityError({
                functionality:
                  "tool-result parts in assistant messages (should be in tool messages)",
              });
            }
            default: {
              const exhaustiveCheck: never = part;
              throw new UnsupportedFunctionalityError({
                functionality: `assistant part type: ${(exhaustiveCheck as { type?: string }).type ?? "unknown"}`,
              });
            }
          }
        }

        const toolCallJson = formatToolCallsJson(toolCallParts);
        const contentSegments: string[] = [];

        if (text.trim().length > 0) {
          contentSegments.push(text);
        } else if (text.length > 0) {
          // preserve purely whitespace responses so we don't lose formatting
          contentSegments.push(text);
        }

        if (toolCallJson) {
          contentSegments.push(toolCallJson);
        }

        const content =
          contentSegments.length > 0 ? contentSegments.join("\n") : "";

        messages.push({
          role: "assistant",
          content,
        } as LanguageModelMessage);
        break;
      }

      case "tool": {
        const toolParts = message.content as LanguageModelV3ToolResultPart[];
        const results: ToolResult[] = toolParts.map(toToolResult);
        const toolResultsJson = formatToolResults(results);

        messages.push({
          role: "user",
          content: toolResultsJson,
        } as LanguageModelMessage);
        break;
      }

      default: {
        const exhaustiveCheck: never = message;
        throw new Error(
          `Unsupported role: ${(exhaustiveCheck as { role?: string }).role ?? "unknown"}`,
        );
      }
    }
  }

  return { systemMessage, messages };
}
