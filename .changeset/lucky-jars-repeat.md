---
"@browser-ai/transformers-js": patch
"@browser-ai/core": patch
"@browser-ai/web-llm": patch
---

Don't warn about the AI SDK's default `toolChoice: { type: "auto" }`

The AI SDK normalizes an unset `toolChoice` to `{ type: "auto" }` and passes it to the provider on every call, with or without tools, so the "toolChoice is not supported" warning fired on every single request. Auto is what these providers already do, so the restated default is now ignored; an explicit `none` / `required` / `tool` choice still warns.
