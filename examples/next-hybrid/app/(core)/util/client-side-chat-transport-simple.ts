import {
  ChatTransport,
  UIMessage,
  UIMessageChunk,
  streamText,
  convertToModelMessages,
  ChatRequestOptions,
  toUIMessageStream,
} from "ai";
import { browserAI } from "@browser-ai/core";

// This class won't stream back data parts with the download progress if
// the Prompt API model hasn't yet been downloaded
export class SimpleClientSideChatTransport implements ChatTransport<UIMessage> {
  async sendMessages(
    options: {
      chatId: string;
      messages: UIMessage[];
      abortSignal: AbortSignal | undefined;
    } & {
      trigger: "submit-message" | "submit-tool-result" | "regenerate-message";
      messageId: string | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const prompt = await convertToModelMessages(options.messages);

    const result = streamText({
      model: browserAI(),
      messages: prompt,
      abortSignal: options.abortSignal,
    });

    return toUIMessageStream({ stream: result.stream });
  }

  async reconnectToStream(
    options: {
      chatId: string;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk> | null> {
    // AFAIK: Client-side AI doesn't support stream reconnection
    return null;
  }
}
