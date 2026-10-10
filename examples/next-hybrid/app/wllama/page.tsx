"use client";

import { useChat } from "@ai-sdk/react";
import {
  Message,
  MessageAvatar,
  MessageContent,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputModelSelect,
  PromptInputModelSelectContent,
  PromptInputModelSelectItem,
  PromptInputModelSelectTrigger,
  PromptInputModelSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Response } from "@/components/ai-elements/response";
import { Loader } from "@/components/ai-elements/loader";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import { Button } from "@/components/ui/button";
import { RefreshCcw, Copy, CheckIcon, XIcon } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { ModeToggle } from "@/components/ui/mode-toggle";
import {
  checkJSPI,
  checkWebGPU,
  doesBrowserSupportWllama,
  wllama,
  WllamaUIMessage,
} from "@browser-ai/wllama";
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
  UIMessage,
} from "ai";
import { toast } from "sonner";
import { Progress } from "@/components/ui/progress";
import { WllamaChatTransport } from "@/app/wllama/util/wllama-chat-transport";
import { ModelSelector } from "@/components/model-selector";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import {
  Confirmation,
  ConfirmationTitle,
  ConfirmationRequest,
  ConfirmationAccepted,
  ConfirmationRejected,
  ConfirmationActions,
  ConfirmationAction,
} from "@/components/ai-elements/confirmation";

const MODELS = [
  "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
  "hf:unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf",
];

type DeviceSupport = {
  wllama: boolean;
  webgpu: boolean;
  jspi: boolean;
};

function WllamaChat({
  support,
  modelId,
  setModelId,
}: {
  support: DeviceSupport;
  modelId: string;
  setModelId: (modelId: string) => void;
}) {
  const browserSupportsWllama = support.wllama;
  const usesGPU = support.webgpu;
  const useJSPI = support.jspi;
  const [input, setInput] = useState("");

  // wllama runs inference in its own worker; no worker setup needed here.
  // wllama defaults to n_ctx 1024, too small once tool schemas and <think> output are in the prompt
  const model = useMemo(
    () =>
      browserSupportsWllama
        ? wllama(modelId, { loadParams: { n_ctx: 4096 } })
        : null,
    [modelId, browserSupportsWllama],
  );

  // Free the model's memory when switching models or leaving the page
  useEffect(() => {
    return () => {
      void model?.unload();
    };
  }, [model]);

  const chatTransport = useMemo(() => {
    if (model) {
      return new WllamaChatTransport(model); // Client side chat transport
    }
    return new DefaultChatTransport<UIMessage>({
      // server side (api route)
      api: "/api/chat",
    });
  }, [model]);

  const {
    error,
    status,
    sendMessage,
    messages,
    regenerate,
    stop,
    addToolApprovalResponse,
  } = useChat<WllamaUIMessage>({
    transport: chatTransport, // use custom transport
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError(error) {
      toast.error(error.message);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && status === "ready") {
      sendMessage({ text: input });
      setInput("");
    }
  };

  const copyMessageToClipboard = (message: any) => {
    const textContent = message.parts
      .filter((part: any) => part.type === "text")
      .map((part: any) => part.text)
      .join("\n");

    navigator.clipboard.writeText(textContent);
  };

  return (
    <div className="flex flex-col h-[calc(100dvh)] max-w-4xl mx-auto">
      <header>
        <div className="flex items-center justify-between p-4">
          <ModelSelector />
          <ModeToggle />
        </div>
      </header>
      {messages.length === 0 && (
        <div className="flex h-full flex-col items-center justify-center text-center">
          {browserSupportsWllama ? (
            <>
              <p className="text-xs">@browser-ai/wllama demo</p>
              <h1 className="text-lg font-medium">
                Using llama.cpp (wllama) client-side AI model
              </h1>
              {usesGPU ? (
                <p className="text-sm max-w-xs">
                  {useJSPI
                    ? "Running on the GPU (WebGPU + WebAssembly JSPI)"
                    : "Running on the GPU (WebGPU)"}
                </p>
              ) : (
                <p className="text-sm max-w-xs">
                  Running on the CPU (WebAssembly): WebGPU is not available.
                </p>
              )}
            </>
          ) : (
            <>
              <h1 className="text-lg font-medium">Using server-side model</h1>
              <p className="text-sm max-w-xs">
                Your device doesn&apos;t support WebAssembly
              </p>
            </>
          )}
        </div>
      )}
      <Conversation className="flex-1">
        <ConversationContent>
          {messages.map((m, index) => (
            <Message
              from={m.role === "system" ? "assistant" : m.role}
              key={m.id}
            >
              <MessageContent>
                {/* Render parts in chronological order */}
                {m.parts.map((part, partIndex) => {
                  // Handle download progress parts
                  if (part.type === "data-modelDownloadProgress") {
                    // Only show if message is not empty (hiding completed/cleared progress)
                    if (!part.data.message) return null;

                    // Don't show the entire div when actively streaming
                    if (status === "ready") return null;

                    return (
                      <div key={partIndex}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="flex items-center gap-1">
                            <Loader className="size-4 " />
                            {part.data.message}
                          </span>
                        </div>
                        {part.data.status === "downloading" &&
                          part.data.progress !== undefined && (
                            <Progress value={part.data.progress} />
                          )}
                      </div>
                    );
                  }

                  // Handle reasoning
                  if (part.type === "reasoning") {
                    return (
                      <Reasoning
                        key={`${m.id}-${partIndex}`}
                        className="w-full"
                        isStreaming={
                          status === "streaming" &&
                          index === messages.length - 1
                        }
                      >
                        <ReasoningTrigger />
                        <ReasoningContent>{part.text}</ReasoningContent>
                      </Reasoning>
                    );
                  }

                  // Handle tool parts
                  if (part.type.startsWith("tool-")) {
                    // Type guard to ensure part is a ToolUIPart
                    if (!("state" in part)) return null;

                    // Handle tool states that need confirmation UI
                    const needsConfirmation =
                      part.state === "approval-requested" ||
                      part.state === "approval-responded" ||
                      part.state === "output-denied";

                    if (needsConfirmation && "approval" in part) {
                      const toolName = part.type.replace("tool-", "");
                      return (
                        <Tool key={partIndex}>
                          <ToolHeader
                            type={part.type as any}
                            state={part.state}
                          />
                          <ToolContent>
                            {"input" in part && part.input !== undefined && (
                              <ToolInput input={part.input} />
                            )}
                            <Confirmation
                              approval={part.approval ?? null}
                              state={part.state}
                            >
                              <ConfirmationTitle>
                                <ConfirmationRequest>
                                  Allow {toolName} to execute with these
                                  parameters?
                                </ConfirmationRequest>
                                <ConfirmationAccepted>
                                  <CheckIcon className="size-4 text-green-600 dark:text-green-400" />
                                  <span>Accepted</span>
                                </ConfirmationAccepted>
                                <ConfirmationRejected>
                                  <XIcon className="size-4 text-destructive" />
                                  <span>Rejected</span>
                                </ConfirmationRejected>
                              </ConfirmationTitle>
                              <ConfirmationActions>
                                <ConfirmationAction
                                  onClick={() =>
                                    addToolApprovalResponse({
                                      id: part.approval!.id,
                                      approved: false,
                                      reason: "User denied tool execution",
                                    })
                                  }
                                  variant="outline"
                                >
                                  Reject
                                </ConfirmationAction>
                                <ConfirmationAction
                                  onClick={() =>
                                    addToolApprovalResponse({
                                      id: part.approval!.id,
                                      approved: true,
                                    })
                                  }
                                  variant="default"
                                >
                                  Accept
                                </ConfirmationAction>
                              </ConfirmationActions>
                            </Confirmation>
                          </ToolContent>
                        </Tool>
                      );
                    }

                    // Map state values to the expected type
                    const toolState =
                      part.state === "streaming" || part.state === "done"
                        ? "output-available"
                        : part.state || "input-streaming";

                    // Format output as ReactNode
                    const formatOutput = (output: unknown): React.ReactNode => {
                      if (output === undefined || output === null)
                        return undefined;
                      if (typeof output === "string") return output;
                      return (
                        <pre className="text-xs overflow-auto">
                          {JSON.stringify(output, null, 2)}
                        </pre>
                      );
                    };

                    return (
                      <Tool key={partIndex}>
                        <ToolHeader
                          type={part.type as any}
                          state={toolState as any}
                        />
                        <ToolContent>
                          {"input" in part && part.input !== undefined && (
                            <ToolInput input={part.input} />
                          )}
                          {("output" in part || "errorText" in part) && (
                            <ToolOutput
                              output={
                                "output" in part && part.output
                                  ? formatOutput(part.output)
                                  : undefined
                              }
                              errorText={
                                "errorText" in part && part.errorText
                                  ? String(part.errorText)
                                  : undefined
                              }
                            />
                          )}
                        </ToolContent>
                      </Tool>
                    );
                  }

                  // Handle text parts
                  if (part.type === "text") {
                    return <Response key={partIndex}>{part.text}</Response>;
                  }

                  return null;
                })}

                {/* Loading state when tool approval was sent and we're waiting for response */}
                {(m.role === "assistant" || m.role === "system") &&
                  index === messages.length - 1 &&
                  status === "submitted" &&
                  m.parts.some(
                    (part) =>
                      part.type.startsWith("tool-") &&
                      "state" in part &&
                      part.state === "approval-responded",
                  ) && (
                    <div className="flex gap-1 items-center text-gray-500 mt-2">
                      <Loader className="size-4" />
                      Thinking...
                    </div>
                  )}

                {/* Action buttons for assistant messages */}
                {(m.role === "assistant" || m.role === "system") &&
                  index === messages.length - 1 &&
                  status === "ready" && (
                    <div className="flex gap-1 mt-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => copyMessageToClipboard(m)}
                        className="text-muted-foreground hover:text-foreground h-4 w-4 [&_svg]:size-3.5"
                      >
                        <Copy />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => regenerate()}
                        className="text-muted-foreground hover:text-foreground h-4 w-4 [&_svg]:size-3.5"
                      >
                        <RefreshCcw />
                      </Button>
                    </div>
                  )}
              </MessageContent>
              <MessageAvatar name={m.role} src={m.role === "user" ? "" : ""} />
            </Message>
          ))}

          {/* Loading state - only show as separate message if not after tool approval */}
          {status === "submitted" &&
            !messages.some(
              (m, index) =>
                index === messages.length - 1 &&
                (m.role === "assistant" || m.role === "system") &&
                m.parts.some(
                  (part) =>
                    part.type.startsWith("tool-") &&
                    "state" in part &&
                    part.state === "approval-responded",
                ),
            ) && (
              <Message from="assistant">
                <MessageContent>
                  <div className="flex gap-1 items-center text-gray-500">
                    <Loader className="size-4" />
                    Thinking...
                  </div>
                </MessageContent>
                <MessageAvatar name="assistant" src="" />
              </Message>
            )}

          {/* Error state */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="text-red-800 mb-2">An error occurred.</div>
              <Button
                type="button"
                variant="outline"
                onClick={() => regenerate()}
                disabled={status === "streaming" || status === "submitted"}
              >
                Retry
              </Button>
            </div>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="p-4">
        <PromptInput
          onSubmit={handleSubmit}
          className="bg-accent dark:bg-card rounded-lg"
        >
          <PromptInputTextarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="What would you like to know? (Powered by wllama / llama.cpp)"
            minHeight={48}
            maxHeight={164}
            className="bg-accent dark:bg-card"
          />
          <PromptInputToolbar>
            <PromptInputTools>
              <PromptInputModelSelect
                onValueChange={setModelId}
                value={modelId}
                disabled={status === "submitted" || status === "streaming"}
              >
                <PromptInputModelSelectTrigger>
                  <PromptInputModelSelectValue />
                </PromptInputModelSelectTrigger>
                <PromptInputModelSelectContent>
                  {MODELS.map((model) => (
                    <PromptInputModelSelectItem key={model} value={model}>
                      {model}
                    </PromptInputModelSelectItem>
                  ))}
                </PromptInputModelSelectContent>
              </PromptInputModelSelect>
            </PromptInputTools>
            <PromptInputSubmit
              disabled={status === "ready" && !input.trim()}
              status={status}
              onClick={
                status === "submitted" || status === "streaming"
                  ? stop
                  : undefined
              }
              type={
                status === "submitted" || status === "streaming"
                  ? "button"
                  : "submit"
              }
            />
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  );
}

export default function WllamaChatPage() {
  const [support, setSupport] = useState<DeviceSupport | null>(null);
  const [modelId, setModelId] = useState(MODELS[0]);

  useEffect(() => {
    setSupport({
      wllama: doesBrowserSupportWllama(),
      webgpu: checkWebGPU(),
      jspi: checkJSPI(),
    });
  }, []);

  if (support === null) {
    return (
      <div className="flex flex-col h-[calc(100dvh)] items-center justify-center max-w-4xl mx-auto">
        <Loader className="size-4" />
      </div>
    );
  }

  return (
    <WllamaChat
      support={support}
      modelId={modelId}
      setModelId={setModelId}
      key={modelId}
    />
  );
}
