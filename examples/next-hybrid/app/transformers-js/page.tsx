"use client";

import { useChat } from "@ai-sdk/react";
import {
  Message,
  MessageAvatar,
  MessageContent,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputButton,
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
import { PlusIcon, RefreshCcw, Copy, X, CheckIcon, XIcon } from "lucide-react";
import { useState, useEffect, useRef, useMemo } from "react";
import { ModeToggle } from "@/components/ui/mode-toggle";
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
  UIMessage,
} from "ai";
import { toast } from "sonner";
import Image from "next/image";
import { Progress } from "@/components/ui/progress";
import { AudioFileDisplay } from "@/components/audio-file-display";
import { ModelSelector } from "@/components/model-selector";
import { TransformersChatTransport } from "./util/transformers-chat-transport";
import {
  runWorkerBenchmark,
  type WorkerBenchmarkResult,
} from "./util/worker-benchmark";
import {
  doesBrowserSupportTransformersJS,
  transformersJS,
  TransformersUIMessage,
} from "@browser-ai/transformers-js";
import { ModelConfig, MODELS } from "./util/models-config";
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

function TransformersJSChat({
  useClientSideInference,
  setUseClientSideInference,
  modelConfig,
  setModelConfig,
}: {
  useClientSideInference: boolean;
  setUseClientSideInference: (value: boolean) => void;
  modelConfig: ModelConfig;
  setModelConfig: (modelConfig: ModelConfig) => void;
}) {
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileList | undefined>(undefined);
  const [isBenchmarkRunning, setIsBenchmarkRunning] = useState(false);
  const [benchmarkStatus, setBenchmarkStatus] = useState<string | null>(null);
  const [benchmarkResult, setBenchmarkResult] =
    useState<WorkerBenchmarkResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canRunBenchmark =
    useClientSideInference &&
    modelConfig.supportsWorker &&
    !modelConfig.isVisionModel;

  const chatTransport = useMemo(() => {
    if (useClientSideInference) {
      const { id, name, supportsWorker, ...modelOptions } = modelConfig;

      const model = transformersJS(modelConfig.id, {
        ...modelOptions,
        worker: supportsWorker
          ? new Worker(new URL("./util/worker.ts", import.meta.url), {
              type: "module",
            })
          : undefined,
      });
      return new TransformersChatTransport(model); // Client side chat transport
    }
    return new DefaultChatTransport<UIMessage>({
      // server side (api route)
      api: "/api/transformers-chat",
    });
  }, [modelConfig, useClientSideInference]);

  const {
    error,
    status,
    sendMessage,
    messages,
    regenerate,
    stop,
    addToolApprovalResponse,
  } = useChat<TransformersUIMessage>({
    transport: chatTransport, // use custom transport
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError(error) {
      toast.error(error.message);
    },
    experimental_throttle: 50,
    id: `chat-${useClientSideInference ? "client" : "server"}-${useClientSideInference ? modelConfig.id : "default"}`, // Forces state refresh (not necessary but fine for this minimal example app)
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
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

  const handleRunBenchmark = async () => {
    if (!canRunBenchmark || isBenchmarkRunning) {
      return;
    }

    setIsBenchmarkRunning(true);
    setBenchmarkResult(null);
    setBenchmarkStatus("Preparing worker benchmark...");

    try {
      const result = await runWorkerBenchmark(modelConfig, setBenchmarkStatus);
      setBenchmarkResult(result);
      setBenchmarkStatus("Benchmark complete");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Benchmark failed";
      setBenchmarkStatus(message);
      toast.error(message);
    } finally {
      setIsBenchmarkRunning(false);
    }
  };

  useEffect(() => {
    setBenchmarkResult(null);
    setBenchmarkStatus(null);
  }, [modelConfig.id]);

  return (
    <div className="flex flex-col h-[calc(100dvh)] max-w-4xl mx-auto">
      <header>
        <div className="flex items-center justify-between p-4">
          <ModelSelector />
          <ModeToggle />
        </div>
        {canRunBenchmark && (
          <div className="px-4 pb-2">
            <div className="rounded-lg border p-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="font-medium">Worker Perf Benchmark</p>
                  <p className="text-muted-foreground">
                    Runs a fixed 6-turn script and reports TTFT/latency + KV
                    cache reuse.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleRunBenchmark}
                  disabled={isBenchmarkRunning}
                >
                  {isBenchmarkRunning ? "Running..." : "Run Benchmark"}
                </Button>
              </div>

              {benchmarkStatus && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {benchmarkStatus}
                </p>
              )}

              {benchmarkResult && (
                <div className="mt-3 space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                    <div className="rounded border p-2">
                      Avg TTFT: {benchmarkResult.summary.avgTtftMs.toFixed(1)}ms
                    </div>
                    <div className="rounded border p-2">
                      Avg Total: {benchmarkResult.summary.avgTotalMs.toFixed(1)}
                      ms
                    </div>
                    <div className="rounded border p-2">
                      Avg t/s:{" "}
                      {benchmarkResult.summary.avgTokensPerSecond.toFixed(2)}
                    </div>
                    <div className="rounded border p-2">
                      Cache reuse:{" "}
                      {(benchmarkResult.summary.cacheReuseRate * 100).toFixed(0)}
                      %
                    </div>
                  </div>
                  <div className="overflow-auto rounded border">
                    <table className="w-full text-xs">
                      <thead className="bg-muted/40">
                        <tr>
                          <th className="px-2 py-1 text-left">Turn</th>
                          <th className="px-2 py-1 text-left">TTFT</th>
                          <th className="px-2 py-1 text-left">Total</th>
                          <th className="px-2 py-1 text-left">In</th>
                          <th className="px-2 py-1 text-left">Out</th>
                          <th className="px-2 py-1 text-left">KV</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchmarkResult.turns.map((turn) => (
                          <tr key={turn.turn} className="border-t">
                            <td className="px-2 py-1">{turn.turn}</td>
                            <td className="px-2 py-1">
                              {turn.ttftMs !== null
                                ? `${turn.ttftMs.toFixed(1)}ms`
                                : "n/a"}
                            </td>
                            <td className="px-2 py-1">
                              {turn.totalMs.toFixed(1)}ms
                            </td>
                            <td className="px-2 py-1">{turn.inputTokens}</td>
                            <td className="px-2 py-1">{turn.outputTokens}</td>
                            <td className="px-2 py-1">
                              {turn.usedPastKeyValues ? "yes" : "no"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </header>
      {messages.length === 0 && (
        <div className="flex h-full flex-col items-center justify-center text-center">
          {useClientSideInference ? (
            <>
              <p className="text-xs">@browser-ai/transformers-js demo</p>
              <h1 className="text-lg font-medium">
                Using Transformers.js client-side
              </h1>
              <p className="text-sm max-w-xs">
                Switch model at the bottom of this page
              </p>
            </>
          ) : (
            <>
              <p className="text-xs">@browser-ai/transformers-js demo</p>
              <h1 className="text-lg font-medium">
                Using server-side TransformersJS
              </h1>
              <p className="text-sm max-w-xs">
                Your device doesn&apos;t support client side inference
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
            placeholder={
              useClientSideInference
                ? "What would you like to know? (Powered by TransformersJS)"
                : "What would you like to know? (Powered by Server-side TransformersJS)"
            }
            minHeight={48}
            maxHeight={164}
            className="bg-accent dark:bg-card"
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
              <PromptInputModelSelect
                disabled={!useClientSideInference}
                onValueChange={(modelId) => {
                  const selectedModel = MODELS.find((m) => m.id === modelId);
                  if (selectedModel) setModelConfig(selectedModel);
                }}
                value={modelConfig.id}
              >
                <PromptInputModelSelectTrigger>
                  <PromptInputModelSelectValue />
                </PromptInputModelSelectTrigger>
                <PromptInputModelSelectContent>
                  {MODELS.map((model) => (
                    <PromptInputModelSelectItem key={model.id} value={model.id}>
                      {model.name}
                    </PromptInputModelSelectItem>
                  ))}
                </PromptInputModelSelectContent>
              </PromptInputModelSelect>
            </PromptInputTools>
            <PromptInputSubmit
              disabled={
                status === "ready" &&
                !input.trim() &&
                (!files || files.length === 0)
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

export default function TransformersJSChatPage() {
  const [useClientSideInference, setUseClientSideInference] = useState<
    boolean | null
  >(null);
  const [modelConfig, setModelConfig] = useState<ModelConfig>(MODELS[0]);

  useEffect(() => {
    setUseClientSideInference(doesBrowserSupportTransformersJS());
  }, []);

  if (useClientSideInference === null) {
    return (
      <div className="flex flex-col h-[calc(100dvh)] items-center justify-center max-w-4xl mx-auto">
        <Loader className="size-4" />
      </div>
    );
  }

  return (
    <TransformersJSChat
      useClientSideInference={useClientSideInference}
      setUseClientSideInference={setUseClientSideInference}
      modelConfig={modelConfig}
      setModelConfig={setModelConfig}
    />
  );
}
