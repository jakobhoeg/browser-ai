import {
  ChatTransport,
  UIMessageChunk,
  streamText,
  convertToModelMessages,
  ChatRequestOptions,
  createUIMessageStream,
  tool,
  stepCountIs,
} from "ai";
import {
  browserAI,
  BrowserAIChatLanguageModel,
  BrowserAIUIMessage,
} from "@browser-ai/core";
import z from "zod";

export const createTools = () => ({
  webSearch: tool({
    description:
      "Search the web for information when you need up-to-date information or facts not in your knowledge base. Use this when the user asks about current events, recent developments, or specific factual information you're unsure about.",
    inputSchema: z.object({
      query: z
        .string()
        .describe("The search query to find information on the web"),
    }),
    needsApproval: true,
    execute: async ({ query }) => {
      try {
        // Call the API route instead of Exa directly
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
 * Options for configuring the ClientSideChatTransport
 */
export interface ClientSideChatTransportOptions {
  /**
   * Callback invoked when the model quota is exceeded.
   * @param event
   */
  onQuotaOverflow?: (event: Event) => void;
}

/**
 * Client-side chat transport AI SDK implementation that handles AI model communication
 * with in-browser AI capabilities.
 *
 * @implements {ChatTransport<BrowserAIUIMessage>}
 */
export class ClientSideChatTransport implements ChatTransport<BrowserAIUIMessage> {
  private tools: ReturnType<typeof createTools>;
  private onQuotaOverflow?: (event: Event) => void;
  private model: BrowserAIChatLanguageModel;

  constructor(options: ClientSideChatTransportOptions = {}) {
    this.tools = createTools();
    this.onQuotaOverflow = options.onQuotaOverflow;
    this.model = browserAI("text", {
      expectedInputs: [{ type: "text" }, { type: "image" }, { type: "audio" }],
      onQuotaOverflow: this.onQuotaOverflow,
    });
  }

  public getInputUsage(): number | undefined {
    return this.model.getInputUsage();
  }

  public getInputQuota(): number | undefined {
    return this.model.getInputQuota();
  }

  async sendMessages(
    options: {
      chatId: string;
      messages: BrowserAIUIMessage[];
      abortSignal: AbortSignal | undefined;
    } & {
      trigger: "submit-message" | "submit-tool-result" | "regenerate-message";
      messageId: string | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const { messages, abortSignal } = options;
    const prompt = await convertToModelMessages(messages);

    return createUIMessageStream<BrowserAIUIMessage>({
      execute: async ({ writer }) => {
        let downloadProgressId: string | undefined;
        const availability = await this.model.availability();

        // Only track progress if model needs downloading
        if (availability !== "available") {
          await this.model.createSessionWithProgress((progress) => {
            const percent = Math.round(progress * 100);

            if (progress >= 1) {
              if (downloadProgressId) {
                writer.write({
                  type: "data-modelDownloadProgress",
                  id: downloadProgressId,
                  data: {
                    status: "complete",
                    progress: 100,
                    message:
                      "Model finished downloading! Getting ready for inference...",
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
                message: `Downloading browser AI model... ${percent}%`,
              },
              transient: !downloadProgressId, // transient only on first write
            });
          });
        }

        // Single streamText call for both paths
        const result = streamText({
          model: this.model,
          tools: this.tools,
          stopWhen: stepCountIs(5),
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

        writer.merge(result.toUIMessageStream({ sendStart: false }));
      },
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
