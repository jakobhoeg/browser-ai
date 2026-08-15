# @browser-ai/core

## 2.1.14

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

## 2.1.13

### Patch Changes

- d7110d2: Use valid SPDX license identifier Apache-2.0

## 2.1.12

### Patch Changes

- 52b76ec: fix: move topK and temperature from per-prompt to session-level options

## 2.1.11

### Patch Changes

- 7b899ab: forward doGenerate abort signals to browser AI

## 2.1.10

### Patch Changes

- fd16f93: omit initialPrompts from BrowserAIChatSettings

## 2.1.9

### Patch Changes

- 41d12ea: Set LanguageModelCreateCoreOptions when calling availability

## 2.1.8

### Patch Changes

- 3649282: chore: add jsDelivr to readme

## 2.1.7

### Patch Changes

- a03d4bd: refactor: update `@types/dom-chromium-ai`

## 2.1.6

### Patch Changes

- 1eae1e9: renamed input usage

## 2.1.5

### Patch Changes

- b3f53a8: perf: reduce prompt traversals and optimize base64 conversion

## 2.1.4

### Patch Changes

- a34bcea: refactor: extract shared streaming processor to eliminate duplicated doStream logic

## 2.1.3

### Patch Changes

- c309a09: perf: improve stream parsing perf from O(n²) to O(n)

## 2.1.2

### Patch Changes

- 6ee5a61: Added the ability to get input usage and input quota from current session

## 2.1.1

### Patch Changes

- bbe98df: chore: update dependencies

## 2.1.0

### Minor Changes

- acc8791: refactor: unify `createSessionWithProgress` to use a `(progress: number) => void` callback across all packages

## 2.0.4

### Patch Changes

- 0f51e16: refactor: extract shared utilities into internal @browser-ai/shared package

## 2.0.3

### Patch Changes

- 9385bd6: fix: use initialPrompts for system prompts per Prompt API spec

## 2.0.2

### Patch Changes

- f8b6996: fix: correct ESM export paths to use .mjs extension

## 2.0.1

### Patch Changes

- 3f665ca: chore: include hero image to npm readme

## 2.0.0

### Major Changes

- 0266287: feat: move package to @browser-ai org
