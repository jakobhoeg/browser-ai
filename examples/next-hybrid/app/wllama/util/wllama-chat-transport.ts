import {
  ChatTransport,
  UIMessageChunk,
  streamText,
  convertToModelMessages,
  ChatRequestOptions,
  createUIMessageStream,
  wrapLanguageModel,
  extractReasoningMiddleware,
  tool,
  isStepCount,
  toUIMessageStream,
} from "ai";
import { WllamaUIMessage, WllamaLanguageModel } from "@browser-ai/wllama";
import z from "zod";

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export const createTools = () => ({
  webSearch: tool({
    description:
      "Search the web for information when you need up-to-date information or facts not in your knowledge base. Use this when the user asks about current events, recent developments, or specific factual information you're unsure about.",
    inputSchema: z.object({
      query: z
        .string()
        .describe("The search query to find information on the web"),
    }),
    execute: async ({ query }) => {
      try {
        const response = await fetch("/api/web-search", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          return errorData.error || "Failed to search the web";
        }

        const result = await response.json();
        return result;
      } catch (err) {
        return `Failed to search the web: ${err instanceof Error ? err.message : "Unknown error"}`;
      }
    },
  }),
  getCurrentTime: tool({
    description:
      "Get the current date and time. Use this when the user asks about the current time, date, or day of the week.",
    inputSchema: z.object({}),
    execute: async () => {
      const now = new Date();
      return {
        timestamp: now.toISOString(),
        date: now.toLocaleDateString("en-US", {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
        }),
        time: now.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: true,
        }),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };
    },
  }),
});

/**
 * Client-side chat transport AI SDK implementation that handles AI model communication
 * with in-browser AI capabilities (wllama / llama.cpp on WebGPU or CPU).
 *
 * @implements {ChatTransport<WllamaUIMessage>}
 */
export class WllamaChatTransport implements ChatTransport<WllamaUIMessage> {
  private readonly model: WllamaLanguageModel;
  private tools: ReturnType<typeof createTools>;

  constructor(model: WllamaLanguageModel) {
    this.model = model;
    this.tools = createTools();
  }

  async sendMessages(
    options: {
      chatId: string;
      messages: WllamaUIMessage[];
      abortSignal: AbortSignal | undefined;
    } & {
      trigger: "submit-message" | "submit-tool-result" | "regenerate-message";
      messageId: string | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const { messages, abortSignal } = options;
    const prompt = await convertToModelMessages(messages);
    const model = this.model;

    return createUIMessageStream<WllamaUIMessage>({
      execute: async ({ writer }) => {
        let downloadProgressId: string | undefined;
        const availability = await model.availability();

        // Only track progress if model needs downloading
        if (availability !== "available") {
          await model.createSessionWithProgress((progress) => {
            const percent = Math.round(progress * 100);

            if (progress >= 1) {
              if (downloadProgressId) {
                writer.write({
                  type: "data-modelDownloadProgress",
                  id: downloadProgressId,
                  data: {
                    status: "complete",
                    progress: 100,
                    message: "Model finished downloading! Loading the model...",
                  },
                });
              }
              return;
            }

            if (!downloadProgressId) {
              downloadProgressId = `download-${Date.now()}`;
            }

            writer.write({
              type: "data-modelDownloadProgress",
              id: downloadProgressId,
              data: {
                status: "downloading",
                progress: percent,
                message: `Downloading GGUF model... ${percent}%`,
              },
              transient: !downloadProgressId, // transient only on first write
            });
          });
        }

        const result = streamText({
          model: wrapLanguageModel({
            model,
            middleware: extractReasoningMiddleware({
              tagName: "think",
            }),
          }),
          tools: this.tools,
          stopWhen: isStepCount(5),
          toolApproval: {
            webSearch: "user-approval",
          },
          messages: prompt,
          abortSignal,
          onChunk: (event) => {
            if (event.chunk.type === "text-delta" && downloadProgressId) {
              writer.write({
                type: "data-modelDownloadProgress",
                id: downloadProgressId,
                data: { status: "complete", progress: 100, message: "" },
              });
              downloadProgressId = undefined;
            }
          },
        });

        writer.merge(
          toUIMessageStream({
            stream: result.stream,
            tools: this.tools,
            sendStart: false,
            onError: getErrorMessage,
          }),
        );
      },
      // Inference runs in the browser, so there are no server details to hide
      onError: getErrorMessage,
    });
  }

  async reconnectToStream(
    options: {
      chatId: string;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk> | null> {
    // Client-side AI doesn't support stream reconnection
    return null;
  }
}
