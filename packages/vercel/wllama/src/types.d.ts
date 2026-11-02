import { UIMessage } from "ai";

/**
 * UI message type for Browser AI features with custom data parts.
 *
 * Extends base UIMessage to include specific data part schemas
 * such as model download progress
 *
 * @example
 * // Import and use with useChat hook from @ai-sdk/react
 * ```typescript
 * import { useChat } from "@ai-sdk/react";
 * import { WllamaUIMessage } from "@browser-ai/wllama";
 *
 * const { messages, sendMessage } = useChat<WllamaUIMessage>({
 *   onData: (dataPart) => {
 *     if (dataPart.type === 'data-modelDownloadProgress') {
 *       console.log(`Download: ${dataPart.data.progress}%`);
 *     }
 *   }
 * });
 * ```
 */
export type WllamaUIMessage = UIMessage<
  never, // No custom metadata type
  {
    /**
     * Model download progress data part for tracking browser AI model download status.
     * Used to display download progress bars and status messages to users.
     */
    modelDownloadProgress: {
      /** Current download/initialization status */
      status: "downloading" | "complete" | "error";
      /** Download progress percentage (0-100), undefined for non-downloading states */
      progress?: number;
      /** Human-readable status message to display to users */
      message: string;
    };
    /**
     * User notification data part for displaying temporary messages and alerts.
     * These are typically transient and not persisted in message history.
     */
    notification: {
      /** The notification message text */
      message: string;
      /** Notification severity level for styling and priority */
      level: "info" | "warning" | "error";
    };
  }
>;

export type Availability =
  | "unavailable"
  | "downloadable"
  | "downloading"
  | "available";
