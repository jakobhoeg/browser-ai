"use client";

import { useChat } from "@ai-sdk/react";
import { ClientSideChatTransport } from "@/app/(core)/util/client-side-chat-transport";
import {
  Message,
  MessageAvatar,
  MessageContent,
} from "@/components/ai-elements/message";
import {
  Confirmation,
  ConfirmationTitle,
  ConfirmationRequest,
  ConfirmationAccepted,
  ConfirmationRejected,
  ConfirmationActions,
  ConfirmationAction,
} from "@/components/ai-elements/confirmation";
import {
  PromptInput,
  PromptInputButton,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import {
  Context,
  ContextContent,
  ContextContentHeader,
  ContextTrigger,
} from "@/components/ai-elements/context";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Response } from "@/components/ai-elements/response";
import { Loader } from "@/components/ai-elements/loader";
import { Button } from "@/components/ui/button";
import {
  PlusIcon,
  MicIcon,
  GlobeIcon,
  RefreshCcw,
  Copy,
  X,
  CheckIcon,
  XIcon,
} from "lucide-react";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { doesBrowserSupportBrowserAI } from "@browser-ai/core";
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
  UIMessage,
} from "ai";
import { toast } from "sonner";
import { BrowserAIUIMessage } from "@browser-ai/core";
import Image from "next/image";
import { Progress } from "@/components/ui/progress";
import { AudioFileDisplay } from "@/components/audio-file-display";
import { Kbd, KbdKey } from "@/components/ui/kbd";
import { ModelSelector } from "@/components/model-selector";
import { SiGithub } from "@icons-pack/react-simple-icons";
import Link from "next/link";
import { BrowserSupportInstructions } from "@/components/browser-support-instructions";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

const doesBrowserSupportModel = doesBrowserSupportBrowserAI();

export default function Chat() {
  const [browserSupportsModel, setBrowserSupportsModel] = useState<
    boolean | null
  >(null);
  const [isClient, setIsClient] = useState(false);

  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileList | undefined>(undefined);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [quotaOverflow, setQuotaOverflow] = useState(false);
  const [inputUsage, setInputUsage] = useState<number | undefined>(undefined);
  const [inputQuota, setInputQuota] = useState<number | undefined>(undefined);

  const clientTransport = useMemo(
    () =>
      doesBrowserSupportModel
        ? new ClientSideChatTransport({
            onQuotaOverflow: () => setQuotaOverflow(true),
          })
        : null,
    [],
  );
  const transport = useMemo(
    () =>
      clientTransport ??
      new DefaultChatTransport<UIMessage>({
        api: "/api/chat",
      }),
    [clientTransport],
  );

  // Check browser support only on client side
  useEffect(() => {
    setIsClient(true);
    setBrowserSupportsModel(doesBrowserSupportBrowserAI());
  }, []);

  const {
    error,
    status,
    sendMessage,
    messages,
    regenerate,
    stop,
    addToolApprovalResponse,
  } = useChat<BrowserAIUIMessage>({
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError(error) {
      toast.error(error.message);
    },
    onData: (dataPart) => {
      // Handle transient notifications
      // we can also access the date-modelDownloadProgress here
      if (dataPart.type === "data-notification") {
        if (dataPart.data.level === "error") {
          toast.error(dataPart.data.message);
        } else if (dataPart.data.level === "warning") {
          toast.warning(dataPart.data.message);
        } else {
          toast.info(dataPart.data.message);
        }
      }
    },
    experimental_throttle: 75,
  });

  const syncInputContext = useCallback(() => {
    if (!clientTransport) return;
    setInputUsage(clientTransport.getInputUsage());
    setInputQuota(clientTransport.getInputQuota());
  }, [clientTransport]);

  useEffect(() => {
    syncInputContext();
  }, [messages, status, syncInputContext]);

  useEffect(() => {
    if (
      !clientTransport ||
      (status !== "submitted" && status !== "streaming")
    ) {
      return;
    }

    const intervalId = window.setInterval(syncInputContext, 250);
    return () => window.clearInterval(intervalId);
  }, [clientTransport, status, syncInputContext]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (quotaOverflow) return;
    if ((input.trim() || files) && status === "ready") {
      sendMessage({
        text: input,
        files,
      });
      setInput("");
      setFiles(undefined);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(e.target.files);
    }
  };

  const removeFile = (indexToRemove: number) => {
    if (files) {
      const dt = new DataTransfer();
      Array.from(files).forEach((file, index) => {
        if (index !== indexToRemove) {
          dt.items.add(file);
        }
      });
      setFiles(dt.files);

      if (fileInputRef.current) {
        fileInputRef.current.files = dt.files;
      }
    }
  };

  const copyMessageToClipboard = (message: any) => {
    const textContent = message.parts
      .filter((part: any) => part.type === "text")
      .map((part: any) => part.text)
      .join("\n");

    navigator.clipboard.writeText(textContent);
  };

  // Show loading state until client-side check completes
  if (!isClient) {
    return (
      <div className="flex flex-col h-[calc(100dvh)] items-center justify-center max-w-4xl mx-auto">
        <Loader className="size-4" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100dvh)] max-w-4xl mx-auto">
      <header>
        <div className="flex items-center justify-between p-4">
          <ModelSelector />
          <div className="flex gap-2 items-center">
            <Link
              href="https://github.com/jakobhoeg/browser-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:opacity-80 transition-opacity"
            >
              <SiGithub />
            </Link>
            <ModeToggle />
          </div>
        </div>
      </header>
      {quotaOverflow && (
        <Alert variant="destructive" className="mx-4 mt-4">
          <AlertTitle>Model quota exceeded</AlertTitle>
          <AlertDescription>
            Start a new chat by refreshing the page.
          </AlertDescription>
        </Alert>
      )}
      {messages.length === 0 && (
        <>
          {browserSupportsModel ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <p className="text-xs">@browser-ai/core demo</p>
              <h1 className="text-lg font-medium">
                Using your browser's AI model
              </h1>
              <p className="text-sm max-w-xs">
                Your browser supports browser AI models
              </p>
            </div>
          ) : (
            <BrowserSupportInstructions />
          )}
        </>
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

                  // Handle file parts
                  if (part.type === "file") {
                    if (part.mediaType?.startsWith("image/")) {
                      return (
                        <div key={partIndex} className="mt-2">
                          <Image
                            src={part.url}
                            width={300}
                            height={300}
                            alt={part.filename || "Uploaded image"}
                            className="object-contain max-w-sm rounded-lg border"
                          />
                        </div>
                      );
                    }

                    if (part.mediaType?.startsWith("audio/")) {
                      return (
                        <AudioFileDisplay
                          key={partIndex}
                          fileName={part.filename!}
                          fileUrl={part.url}
                        />
                      );
                    }

                    // TODO: Handle other file types
                    return null;
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
            placeholder="What would you like to know?"
            minHeight={48}
            maxHeight={164}
            className="bg-accent dark:bg-card"
            disabled={quotaOverflow}
          />
          <PromptInputToolbar>
            <PromptInputTools>
              <PromptInputButton onClick={() => fileInputRef.current?.click()}>
                <PlusIcon size={16} />
              </PromptInputButton>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple
                accept="image/*,text/*,audio/*"
                className="hidden"
              />
              <PromptInputButton>
                <MicIcon size={16} />
              </PromptInputButton>
              <PromptInputButton>
                <GlobeIcon size={16} />
                <span>Search</span>
              </PromptInputButton>
            </PromptInputTools>
            <div className="flex items-center gap-2">
              {inputQuota !== undefined && inputQuota > 0 && (
                <Context usedTokens={inputUsage ?? 0} maxTokens={inputQuota}>
                  <ContextTrigger className="h-8 px-2.5" />
                  <ContextContent align="end">
                    <ContextContentHeader />
                  </ContextContent>
                </Context>
              )}
              <Kbd>
                <KbdKey aria-label="Control">Ctrl</KbdKey>
                <KbdKey>Enter</KbdKey>
              </Kbd>
              <PromptInputSubmit
                disabled={
                  quotaOverflow ||
                  (status === "ready" &&
                    !input.trim() &&
                    (!files || files.length === 0))
                }
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
            </div>
          </PromptInputToolbar>

          {/* File preview area - moved inside the form */}
          {files && files.length > 0 && (
            <div className="w-full flex px-2 p-2 gap-2">
              {Array.from(files).map((file, index) => (
                <div
                  key={index}
                  className="relative bg-muted-foreground/20 flex w-fit flex-col gap-2 p-1 border-t border-x rounded-md"
                >
                  {file.type.startsWith("image/") ? (
                    <div className="flex text-sm">
                      <Image
                        width={100}
                        height={100}
                        src={URL.createObjectURL(file)}
                        alt={file.name}
                        className="h-auto rounded-md w-auto max-w-[100px] max-h-[100px]"
                      />
                    </div>
                  ) : file.type.startsWith("audio/") ? (
                    <div className="flex text-sm flex-col">
                      <audio src={URL.createObjectURL(file)} className="hidden">
                        Your browser does not support the audio element.
                      </audio>
                      <span className="text-xs text-gray-500 truncate max-w-[100px]">
                        {file.name}
                      </span>
                    </div>
                  ) : (
                    <div className="flex text-sm">
                      <span className="text-xs truncate max-w-[100px]">
                        {file.name}
                      </span>
                    </div>
                  )}
                  <button
                    onClick={() => removeFile(index)}
                    className="absolute -top-1.5 -right-1.5 text-white cursor-pointer bg-red-500 hover:bg-red-600 w-4 h-4 rounded-full flex items-center justify-center"
                    type="button"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </PromptInput>
      </div>
    </div>
  );
}
