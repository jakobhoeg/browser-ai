# @browser-ai/web-llm

## 2.1.10

### Patch Changes

- d5cf173: Parse structured argument values in Python-style tool calls.

  Argument values were stored as raw text, so a call like
  `[ask_user(questions=[{'question': 'How many?'}])]` produced a string
  containing Python syntax instead of an array. Values are now parsed as JSON or
  Python literals, covering lists, dicts, numbers, and `True`/`False`/`None`.

  Calls whose arguments contain parentheses (`q="budget (roughly)?"`) are also no
  longer truncated at the first `)`.

## 2.1.9

### Patch Changes

- 48f3445: Fix tool call parsing for Python-style and delimiter-wrapped formats.
  - Arguments are no longer split on commas inside quoted values, so a call like
    `search(query="Doe, Jane")` keeps its argument intact.
  - Multiple calls in a single bracket (`[a(...), b(...)]`) are now parsed;
    previously the whole block was skipped.
  - `<|tool_call_start|>` / `<|tool_call_end|>` delimiters are recognized, so
    models using them (LFM2.5) no longer leak the markers into response text.
  - transformers-js: tool call arguments are passed to chat templates as a
    mapping rather than a JSON-encoded string, which some templates reject.

## 2.1.8

### Patch Changes

- d7110d2: Use valid SPDX license identifier Apache-2.0

## 2.1.7

### Patch Changes

- 3649282: chore: add jsDelivr to readme

## 2.1.6

### Patch Changes

- 72b741c: fix: type requestOptions & fix providerOptions key lookup

## 2.1.5

### Patch Changes

- a34bcea: refactor: extract shared streaming processor to eliminate duplicated doStream logic

## 2.1.4

### Patch Changes

- c309a09: perf: improve stream parsing perf from O(n²) to O(n)

## 2.1.3

### Patch Changes

- bbe98df: chore: update dependencies

## 2.1.2

### Patch Changes

- 7bd5bbe: feat: add webllm generation config

## 2.1.1

### Patch Changes

- 8ba3a0e: feat: add embedding model provider with WebLLMEmbeddingModel class

## 2.1.0

### Minor Changes

- acc8791: refactor: unify `createSessionWithProgress` to use a `(progress: number) => void` callback across all packages

## 2.0.4

### Patch Changes

- 0f51e16: refactor: extract shared utilities into internal @browser-ai/shared package

## 2.0.3

### Patch Changes

- f8b6996: fix: correct ESM export paths to use .mjs extension

## 2.0.2

### Patch Changes

- b20ac86: fix: structured output

## 2.0.1

### Patch Changes

- 3f665ca: chore: include hero image to npm readme

## 2.0.0

### Major Changes

- 0266287: feat: move package to @browser-ai org
