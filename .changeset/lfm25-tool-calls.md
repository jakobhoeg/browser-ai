---
"@browser-ai/transformers-js": patch
"@browser-ai/web-llm": patch
"@browser-ai/core": patch
---

Fix tool call parsing for Python-style and delimiter-wrapped formats.

- Arguments are no longer split on commas inside quoted values, so a call like
  `search(query="Doe, Jane")` keeps its argument intact.
- Multiple calls in a single bracket (`[a(...), b(...)]`) are now parsed;
  previously the whole block was skipped.
- `<|tool_call_start|>` / `<|tool_call_end|>` delimiters are recognized, so
  models using them (LFM2.5) no longer leak the markers into response text.
- transformers-js: tool call arguments are passed to chat templates as a
  mapping rather than a JSON-encoded string, which some templates reject.
