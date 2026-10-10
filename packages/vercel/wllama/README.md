# wllama provider for Vercel AI SDK

<div align="center">

[![NPM Version](https://img.shields.io/npm/v/%40browser-ai%2Fwllama)](https://www.npmjs.com/package/@browser-ai/wllama)
[![NPM Downloads](https://img.shields.io/npm/dm/%40browser-ai%2Fwllama)](https://www.npmjs.com/package/@browser-ai/wllama)
[![jsDelivr hits (npm)](https://img.shields.io/jsdelivr/npm/hm/%40browser-ai%2Fwllama)](https://www.jsdelivr.com/package/npm/@browser-ai/wllama)

</div>

[wllama](https://github.com/ngxson/wllama) model provider for [Vercel AI SDK](https://ai-sdk.dev/). This library lets you run any GGUF model with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s WebGPU backend directly in the browser, using the AI SDK you already know.

## Installation

```bash
npm i @browser-ai/wllama @wllama/wllama@3.6.1
```

## Notes

- Model ids accept `hf:owner/repo/path/model.gguf` or a direct `https://…/model.gguf` URL.
- `@wllama/wllama` must be exactly `3.6.1`, matching the WASM binary loaded from jsDelivr.
- GGUF files are limited to 2GB each. Split larger models with `llama-gguf-split` and pass the first shard.
- WebGPU is used when available (except on Firefox); otherwise the model runs on the CPU.

## Author

2025 © Jakob Hoeg Mørk

## Credits

The wllama, llama.cpp & Vercel teams
