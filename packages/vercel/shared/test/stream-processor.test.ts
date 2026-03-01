import { describe, it, expect } from "vitest";
import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import {
  processToolCallStream,
  generateToolCallId,
} from "../src/streaming/stream-processor";

async function* makeChunks(strings: string[]): AsyncIterable<string> {
  for (const s of strings) yield s;
}

function makeController() {
  const events: LanguageModelV3StreamPart[] = [];
  const controller = {
    enqueue: (event: LanguageModelV3StreamPart) => events.push(event),
  } as unknown as ReadableStreamDefaultController<LanguageModelV3StreamPart>;
  return { controller, events };
}

function makeTextCollector() {
  const parts: string[] = [];
  const emitTextDelta = (delta: string) => parts.push(delta);
  return { emitTextDelta, getText: () => parts.join("") };
}

async function run(chunks: string[], options?: { stopEarlyOnToolCall?: boolean }) {
  const { controller, events } = makeController();
  const { emitTextDelta, getText } = makeTextCollector();
  const result = await processToolCallStream(
    makeChunks(chunks),
    emitTextDelta,
    controller,
    options,
  );
  return { result, events, text: getText() };
}

describe("processToolCallStream — plain text", () => {
  it("emits text via emitTextDelta, no tool detection", async () => {
    const { result, text } = await run(["Hello", " world"]);
    expect(result.toolCallDetected).toBe(false);
    expect(text).toBe("Hello world");
  });

  it("handles an empty iterable", async () => {
    const { result, events, text } = await run([]);
    expect(result.toolCallDetected).toBe(false);
    expect(events).toHaveLength(0);
    expect(text).toBe("");
  });
});

describe("processToolCallStream — valid tool call", () => {
  const FENCE = '```tool_call\n{"name": "search", "arguments": {"q": "test"}}\n```';

  it("detects the tool call and returns the parsed result", async () => {
    const { result } = await run([FENCE]);
    expect(result.toolCallDetected).toBe(true);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0].toolName).toBe("search");
    expect(result.toolCalls[0].args).toEqual({ q: "test" });
  });

  it("emits events in order: tool-input-start → tool-input-end → tool-call", async () => {
    const { events } = await run([FENCE]);
    const types = events.map((e) => e.type);
    expect(types).toContain("tool-input-start");
    expect(types.indexOf("tool-input-start")).toBeLessThan(types.indexOf("tool-input-end"));
    expect(types.indexOf("tool-input-end")).toBeLessThan(types.indexOf("tool-call"));
  });

  it("emits a tool-call event with correct fields", async () => {
    const { events } = await run([FENCE]);
    const ev = events.find((e) => e.type === "tool-call") as Extract<
      LanguageModelV3StreamPart,
      { type: "tool-call" }
    >;
    expect(ev.toolName).toBe("search");
    expect(ev.input).toBe(JSON.stringify({ q: "test" }));
    expect(ev.providerExecuted).toBe(false);
  });

  it("uses the same toolCallId across all events for the same call", async () => {
    const { events } = await run([FENCE]);
    const ids = events
      .filter((e) => ["tool-input-start", "tool-input-delta", "tool-input-end", "tool-call"].includes(e.type))
      .map((e) => (e.type === "tool-call" ? e.toolCallId : (e as { id: string }).id));
    expect(new Set(ids).size).toBe(1);
    expect(ids[0]).toMatch(/^call_/);
  });
});

describe("processToolCallStream — chunked tool call", () => {
  it("assembles a tool call from multiple chunks", async () => {
    const { result } = await run([
      "```tool_call\n",
      '{"name": "calc", "arguments":',
      ' {"x": 42}}\n',
      "```",
    ]);
    expect(result.toolCallDetected).toBe(true);
    expect(result.toolCalls[0].toolName).toBe("calc");
    expect(result.toolCalls[0].args).toEqual({ x: 42 });
  });

  it("emits tool-input-start as soon as the tool name is available mid-stream", async () => {
    const { events } = await run([
      "```tool_call\n",
      '{"name": "early_name", "arguments": {"k":',
      ' "v"}}\n```',
    ]);
    const startEvent = events.find((e) => e.type === "tool-input-start") as
      | Extract<LanguageModelV3StreamPart, { type: "tool-input-start" }>
      | undefined;
    expect(startEvent).toBeDefined();
    expect(startEvent!.toolName).toBe("early_name");
  });
});

describe("processToolCallStream — text around tool calls", () => {
  it("emits text before the fence via emitTextDelta", async () => {
    const { text } = await run([
      'Before. ```tool_call\n{"name": "t", "arguments": {}}\n```',
    ]);
    expect(text).toContain("Before.");
  });

  it("returns text after the fence as trailingText (not emitted inline)", async () => {
    const { result, text } = await run([
      '```tool_call\n{"name": "t", "arguments": {}}\n``` after',
    ]);
    expect(result.trailingText).toBe(" after");
    expect(text).not.toContain("after");
  });
});

describe("processToolCallStream — malformed fence", () => {
  it("treats a fence with non-JSON content as plain text", async () => {
    const { result, text } = await run(["```tool_call\nnot json\n```"]);
    expect(result.toolCallDetected).toBe(false);
    expect(text).toContain("not json");
  });

  it("treats a fence with no 'name' field as plain text", async () => {
    const { result } = await run(['```tool_call\n{"arguments": {"x": 1}}\n```']);
    expect(result.toolCallDetected).toBe(false);
  });
});

describe("processToolCallStream — stopEarlyOnToolCall", () => {
  function makeTrackedChunks(strings: string[]) {
    const consumed: string[] = [];
    async function* gen() {
      for (const s of strings) {
        consumed.push(s);
        yield s;
      }
    }
    return { gen, consumed };
  }

  const CHUNKS = [
    '```tool_call\n{"name": "t", "arguments": {}}\n```',
    "extra-1",
    "extra-2",
  ];

  it("stops consuming chunks after tool detection when true", async () => {
    const { gen, consumed } = makeTrackedChunks(CHUNKS);
    const { controller } = makeController();
    const { emitTextDelta } = makeTextCollector();
    await processToolCallStream(gen(), emitTextDelta, controller, { stopEarlyOnToolCall: true });
    expect(consumed).not.toContain("extra-1");
  });

  it("drains all chunks after tool detection when false (default)", async () => {
    const { gen, consumed } = makeTrackedChunks(CHUNKS);
    const { controller } = makeController();
    const { emitTextDelta } = makeTextCollector();
    await processToolCallStream(gen(), emitTextDelta, controller);
    expect(consumed).toContain("extra-1");
    expect(consumed).toContain("extra-2");
  });
});

describe("processToolCallStream — edge cases", () => {
  it("only uses the first tool call from a multi-call fence", async () => {
    const { result } = await run([
      '```tool_call\n[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]\n```',
    ]);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0].toolName).toBe("a");
  });
});
