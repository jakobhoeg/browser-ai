import type { ModelConfig } from "./models-config";

const BENCHMARK_PROMPTS = [
  "Write a one-sentence summary of photosynthesis.",
  "Now explain it to a 10-year-old in one sentence.",
  "Give a real-world analogy in one sentence.",
  "List 3 key terms from your previous explanation.",
  "Turn those terms into a short quiz question.",
  "Answer the quiz question yourself in one sentence.",
];

type WorkerStatus =
  | "loading"
  | "ready"
  | "start"
  | "update"
  | "complete"
  | "error";

interface WorkerResponse {
  status: WorkerStatus;
  data?: string;
  output?: string | string[];
  numTokens?: number;
  inputLength?: number;
  usedPastKeyValues?: boolean;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface BenchmarkTurnResult {
  turn: number;
  prompt: string;
  ttftMs: number | null;
  totalMs: number;
  inputTokens: number;
  outputTokens: number;
  tokensPerSecond: number | null;
  usedPastKeyValues: boolean;
}

export interface WorkerBenchmarkResult {
  turns: BenchmarkTurnResult[];
  summary: {
    avgTtftMs: number;
    avgTotalMs: number;
    avgTokensPerSecond: number;
    cacheReuseRate: number;
  };
}

function createBenchmarkWorker() {
  return new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
}

function getAssistantText(output: string | string[] | undefined): string {
  if (Array.isArray(output)) {
    return output[0] ?? "";
  }
  return output ?? "";
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

async function loadWorkerModel(
  worker: Worker,
  modelConfig: ModelConfig,
  onStatus?: (status: string) => void,
) {
  await new Promise<void>((resolve, reject) => {
    const onMessage = (e: MessageEvent<WorkerResponse>) => {
      const message = e.data;
      if (!message || typeof message !== "object") return;

      if (message.status === "loading" && message.data) {
        onStatus?.(message.data);
      } else if (message.status === "ready") {
        worker.removeEventListener("message", onMessage);
        resolve();
      } else if (message.status === "error") {
        worker.removeEventListener("message", onMessage);
        reject(new Error(message.data || "Worker load failed"));
      }
    };

    worker.addEventListener("message", onMessage);
    worker.postMessage({
      type: "load",
      data: {
        modelId: modelConfig.id,
        dtype: modelConfig.dtype,
        device: modelConfig.device,
        use_external_data_format: modelConfig.use_external_data_format,
        isVisionModel: modelConfig.isVisionModel,
      },
    });
  });
}

async function runWorkerTurn(worker: Worker, messages: ChatMessage[]) {
  return await new Promise<{
    assistantText: string;
    ttftMs: number | null;
    totalMs: number;
    inputTokens: number;
    outputTokens: number;
    usedPastKeyValues: boolean;
  }>((resolve, reject) => {
    let startedAt = 0;
    let firstTokenAt: number | null = null;
    let streamedOutput = "";

    const onMessage = (e: MessageEvent<WorkerResponse>) => {
      const message = e.data;
      if (!message || typeof message !== "object") return;

      if (message.status === "start") {
        startedAt = performance.now();
      } else if (message.status === "update") {
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        if (typeof message.output === "string") {
          streamedOutput += message.output;
        }
      } else if (message.status === "complete") {
        worker.removeEventListener("message", onMessage);
        const completedAt = performance.now();
        const output = streamedOutput || getAssistantText(message.output);
        const totalMs = startedAt > 0 ? completedAt - startedAt : 0;
        resolve({
          assistantText: output,
          ttftMs: firstTokenAt !== null && startedAt > 0 ? firstTokenAt - startedAt : null,
          totalMs,
          inputTokens: message.inputLength ?? 0,
          outputTokens: message.numTokens ?? 0,
          usedPastKeyValues: !!message.usedPastKeyValues,
        });
      } else if (message.status === "error") {
        worker.removeEventListener("message", onMessage);
        reject(new Error(message.data || "Worker generation failed"));
      }
    };

    worker.addEventListener("message", onMessage);
    worker.postMessage({
      type: "generate",
      data: messages,
    });
  });
}

export async function runWorkerBenchmark(
  modelConfig: ModelConfig,
  onStatus?: (status: string) => void,
): Promise<WorkerBenchmarkResult> {
  const worker = createBenchmarkWorker();
  const conversation: ChatMessage[] = [];
  const turns: BenchmarkTurnResult[] = [];

  try {
    onStatus?.("Loading model...");
    await loadWorkerModel(worker, modelConfig, onStatus);

    for (let i = 0; i < BENCHMARK_PROMPTS.length; i++) {
      const prompt = BENCHMARK_PROMPTS[i];
      onStatus?.(`Running turn ${i + 1}/${BENCHMARK_PROMPTS.length}...`);

      conversation.push({ role: "user", content: prompt });
      const turn = await runWorkerTurn(worker, conversation);
      conversation.push({ role: "assistant", content: turn.assistantText });

      const tokensPerSecond =
        turn.outputTokens > 0 && turn.totalMs > 0
          ? (turn.outputTokens / turn.totalMs) * 1000
          : null;

      turns.push({
        turn: i + 1,
        prompt,
        ttftMs: turn.ttftMs,
        totalMs: turn.totalMs,
        inputTokens: turn.inputTokens,
        outputTokens: turn.outputTokens,
        tokensPerSecond,
        usedPastKeyValues: turn.usedPastKeyValues,
      });
    }

    const avgTtftMs = average(
      turns
        .map((turn) => turn.ttftMs)
        .filter((value): value is number => value !== null),
    );
    const avgTotalMs = average(turns.map((turn) => turn.totalMs));
    const avgTokensPerSecond = average(
      turns
        .map((turn) => turn.tokensPerSecond)
        .filter((value): value is number => value !== null),
    );

    const reusableTurns = Math.max(turns.length - 1, 1);
    const reusedTurns = turns
      .slice(1)
      .filter((turn) => turn.usedPastKeyValues).length;

    return {
      turns,
      summary: {
        avgTtftMs,
        avgTotalMs,
        avgTokensPerSecond,
        cacheReuseRate: reusedTurns / reusableTurns,
      },
    };
  } finally {
    worker.terminate();
  }
}
