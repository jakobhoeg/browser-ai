# Browser AI model providers for Vercel AI SDK

<div align="center">

<br />

<a href="https://vercel.com/oss">
  <img alt="Vercel OSS Program" src="https://vercel.com/oss/program-badge.svg" />
</a>

<br />
<br />
  
<img src="npm-header.gif">
</div>

<div align="center">

[![NPM Downloads](https://img.shields.io/npm/dm/%40browser-ai%2Fcore)](https://www.npmjs.com/package/@browser-ai/core)
[![NPM Downloads](https://img.shields.io/npm/dm/%40browser-ai%2Fweb-llm)](https://www.npmjs.com/package/@browser-ai/web-llm)
[![NPM Downloads](https://img.shields.io/npm/dm/%40browser-ai%2Ftransformers-js)](https://www.npmjs.com/package/@browser-ai/transformers-js)

</div>

TypeScript libraries that provide access to in-browser AI model providers with seamless fallback to using server-side models.

## Documentation

For detailed documentation, browser requirements and advanced usage, refer to the [official documentation site](https://browser-ai.dev/docs).

### Package Versions

| Package                       | AI SDK v5 |  AI SDK v6  |
| ----------------------------- | :-------: | :---------: |
| `@browser-ai/core`            | ✓ `1.0.0` | ✓ `≥ 2.0.0` |
| `@browser-ai/transformers-js` | ✓ `1.0.0` | ✓ `≥ 2.0.0` |
| `@browser-ai/web-llm`         | ✓ `1.0.0` | ✓ `≥ 2.0.0` |

```bash
# For Chrome/Edge built-in browser AI models
npm i @browser-ai/core

# For open-source models via WebLLM
npm i @browser-ai/web-llm

# For 🤗 Transformers.js models
npm i @browser-ai/transformers-js
```

### Basic Usage with Chrome/Edge AI

```typescript
import { streamText } from "ai";
import { browserAI } from "@browser-ai/core";

const result = streamText({
  model: browserAI(),
  prompt: "Invent a new holiday and describe its traditions.",
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
```

### Basic Usage with WebLLM

```typescript
import { streamText } from "ai";
import { webLLM } from "@browser-ai/web-llm";

const result = streamText({
  model: webLLM("Llama-3.2-3B-Instruct-q4f16_1-MLC"),
  prompt: "Invent a new holiday and describe its traditions.",
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
```

### Basic Usage with Transformers.js

```typescript
import { streamText } from "ai";
import { transformersJS } from "@browser-ai/transformers-js";

const result = streamText({
  model: transformersJS("HuggingFaceTB/SmolLM2-360M-Instruct"),
  prompt: "Invent a new holiday and describe its traditions.",
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
```

## Example applications

- Huggingface Transformers.js AI SDK v5: [next-vercel-ai-sdk-v5-transformers-js-example](https://github.com/huggingface/transformers.js-examples/tree/main/next-vercel-ai-sdk-v5)
- Huggingface Transformers.js AI SDK v6: [next-vercel-ai-sdk-v6-transformers-js-example](https://github.com/huggingface/transformers.js-examples/blob/main/next-vercel-ai-sdk-v6-tool-calling)

## Sponsors

This project is proudly sponsored by [Chrome for Developers](https://developer.chrome.com/) and is part of the [Vercel OSS Program](https://vercel.com/open-source-program).

## Contributing

Contributions are more than welcome! However, please make sure to check out the [contribution guidelines](https://github.com/jakobhoeg/browser-ai/blob/main/CONTRIBUTING.md) before contributing.

## Why?

If you've ever built apps with local language models, you're likely familiar with the challenges: creating custom hooks, UI components and state management (lots of it), while also building complex integration layers to fall back to server-side models when compatibility is an issue.

Read more about this [here](https://www.browser-ai.dev/docs/ai-sdk-v6).

## Author

2025 © Jakob Hoeg Mørk
