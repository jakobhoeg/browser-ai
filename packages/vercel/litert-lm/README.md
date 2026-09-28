# LiteRT-LM provider for Vercel AI SDK

[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (via [LiteRT.js](https://developers.google.com/edge/litert/web)) model provider for the [Vercel AI SDK](https://ai-sdk.dev/). This library lets you run Google's on-device LLMs — starting with **Gemma 4** — directly in the browser with WebGPU, with seamless fallback to server-side models through the AI SDK's provider system.

> **Status:** The LiteRT-LM Web SDK is an early preview (text-in / text-out on WebGPU). Tool/function calling, JSON response format, and multimodal input are not yet exposed by the web runtime and will emit warnings if requested.

## Installation

```bash
npm i @browser-ai/litert-lm
```

`@browser-ai/litert-lm` wraps [`@litert-lm/core`](https://www.npmjs.com/package/@litert-lm/core) and works with AI SDK v7 (`ai@^7`).

## Usage

```typescript
import { streamText } from "ai";
import { liteRTLM, GEMMA_4_E2B_WEB } from "@browser-ai/litert-lm";

const result = streamText({
  model: liteRTLM(GEMMA_4_E2B_WEB),
  prompt: "Explain WebGPU in one sentence.",
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

A model id is the location of a `.litertlm` artifact — an HTTPS URL, a local
path, a `Blob`, or a `ReadableStream`. Convenience constants for the models
supported by the Web SDK are exported from the package root:

- `GEMMA_4_E2B_WEB` — `gemma-4-E2B-it-web.litertlm`
- `GEMMA_4_E4B_WEB` — `gemma-4-E4B-it-web.litertlm`

### Settings

```typescript
liteRTLM(GEMMA_4_E2B_WEB, {
  // Forward raw EngineSettings to Engine.create (e.g. a custom model Blob)
  engineSettings: { mainExecutorSettings: { maxNumTokens: 16384 } },
  // Enable reasoning ("thinking") tokens on reasoning-capable models
  enableThinking: true,
});
```

You can also toggle thinking per-call via provider options:

```typescript
streamText({
  model: liteRTLM(GEMMA_4_E2B_WEB),
  prompt: "Solve this step by step.",
  providerOptions: { "litert-lm": { enableThinking: true } },
});
```

### Checking availability

```typescript
import { liteRTLM, doesBrowserSupportLiteRTLM } from "@browser-ai/litert-lm";

if (doesBrowserSupportLiteRTLM()) {
  const model = liteRTLM(GEMMA_4_E2B_WEB);
  // Optionally warm up the engine (downloads the model):
  await model.createSessionWithProgress((p) => console.log(`${p * 100}%`));
}
```

## Notes & limitations

- Requires a browser with **WebGPU** support (Chrome/Edge). Calling without it throws a `LoadSettingError`.
- The web SDK is currently **text-only**: image/audio file inputs throw `UnsupportedFunctionalityError`.
- Token usage statistics are not yet reported by the web runtime (left as `undefined`).
- The conversation is treated as stateless per AI SDK call: prior turns are replayed via the conversation preface so multi-turn context is preserved.

## Documentation

For full documentation, see the [browser-ai docs](https://www.browser-ai.dev/docs).

## Author

2025 © Jakob Hoeg Mørk

## Credits

The Vercel, Google AI Edge, and browser-ai teams.
