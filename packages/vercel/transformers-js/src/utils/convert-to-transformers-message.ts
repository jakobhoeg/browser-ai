import {
  LanguageModelV3Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";

function uint8ArrayToBase64(uint8array: Uint8Array): string {
  return btoa(String.fromCharCode(...uint8array));
}

function convertDataToURL(
  data:
    | string
    | Buffer
    | URL
    | Uint8Array
    | ArrayBuffer
    | ReadableStream
    | undefined,
  mediaType: string,
): string {
  if (data instanceof URL) return data.toString();

  if (typeof data === "string") {
    return `data:${mediaType};base64,${data}`;
  }

  if (data instanceof Uint8Array) {
    return `data:${mediaType};base64,${uint8ArrayToBase64(data)}`;
  }

  if (data instanceof ArrayBuffer) {
    return `data:${mediaType};base64,${uint8ArrayToBase64(new Uint8Array(data))}`;
  }

  if (typeof Buffer !== "undefined" && data instanceof Buffer) {
    return `data:${mediaType};base64,${data.toString("base64")}`;
  }

  throw new UnsupportedFunctionalityError({
    functionality: `file data type: ${typeof data}`,
  });
}

/**
 * TransformersJS message type compatible with HuggingFace chat templates
 */
export interface TransformersMessage {
  role: string;
  content:
    | string
    | null
    | Array<{ type: string; text?: string; image?: string }>;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  name?: string;
}

/**
 * Safely normalize tool arguments - handles both string and object inputs
 */
function normalizeToolArguments(input: unknown): unknown {
  if (input === undefined) {
    return {};
  }

  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      // If parsing fails, return the string as-is
      return input;
    }
  }

  return input ?? {};
}

function processVisionContent(
  content: any[],
): Array<{ type: "text"; text: string } | { type: "image"; image: string }> {
  const contentParts: Array<
    { type: "text"; text: string } | { type: "image"; image: string }
  > = [];
  let textParts: string[] = [];

  for (const part of content) {
    if (part.type === "text") {
      textParts.push(part.text);
    } else if (part.type === "file" && part.mediaType?.startsWith("image/")) {
      if (textParts.length > 0) {
        contentParts.push({ type: "text", text: textParts.join("\n") });
        textParts = [];
      }
      contentParts.push({
        type: "image",
        image: convertDataToURL(part.data, part.mediaType!),
      });
    } else if (part.type === "file") {
      throw new UnsupportedFunctionalityError({
        functionality: "non-image file input",
      });
    }
  }

  if (textParts.length > 0) {
    contentParts.push({ type: "text", text: textParts.join("\n") });
  }

  return contentParts;
}

export function convertToTransformersMessages(
  prompt: LanguageModelV3Prompt,
  isVisionModel: boolean = false,
): TransformersMessage[] {
  return prompt.flatMap(
    (message): TransformersMessage | TransformersMessage[] => {
      switch (message.role) {
        case "system":
          return { role: "system", content: message.content };

        case "user":
          if (isVisionModel) {
            return {
              role: "user",
              content: processVisionContent(message.content),
            };
          }

          const textContent = message.content
            .map((part) => {
              if (part.type === "text") return part.text;
              if (part.type === "file")
                throw new UnsupportedFunctionalityError({
                  functionality: "file input",
                });
              return "";
            })
            .join("\n");
          return { role: "user", content: textContent };

        case "assistant":
          const textParts = message.content
            .filter((part) => part.type === "text")
            .map((part) => (part as any).text);

          const toolCallParts = message.content.filter(
            (part) => part.type === "tool-call",
          );

          // If there are tool calls, format as HuggingFace expects
          if (toolCallParts.length > 0) {
            const tool_calls = toolCallParts.map((part) => ({
              id: (part as any).toolCallId,
              type: "function" as const,
              function: {
                name: (part as any).toolName,
                arguments:
                  typeof (part as any).input === "string"
                    ? (part as any).input
                    : JSON.stringify(
                        normalizeToolArguments((part as any).input),
                      ),
              },
            }));

            return {
              role: "assistant",
              content: textParts.length > 0 ? textParts.join("\n") : null,
              tool_calls,
            };
          }

          return {
            role: "assistant",
            content: textParts.join("\n"),
          };

        case "tool":
          // Each tool result becomes a separate message with role "tool"
          // This is the native HuggingFace format for tool results
          return message.content
            .filter((part) => part.type === "tool-result")
            .map((part) => {
              const toolPart = part as any;

              let resultValue: unknown;
              switch (toolPart.output.type) {
                case "text":
                case "json":
                case "content":
                  resultValue = toolPart.output.value;
                  break;
                case "error-text":
                case "error-json":
                  resultValue = {
                    error: true,
                    message: toolPart.output.value,
                  };
                  break;
                case "execution-denied":
                  resultValue = {
                    error: true,
                    reason: toolPart.output.reason ?? "execution denied",
                  };
                  break;
              }

              return {
                role: "tool",
                tool_call_id: toolPart.toolCallId,
                name: toolPart.toolName,
                content:
                  typeof resultValue === "string"
                    ? resultValue
                    : JSON.stringify(resultValue),
              };
            });

        default:
          throw new Error(`Unsupported message role: ${(message as any).role}`);
      }
    },
  );
}
