---
"@browser-ai/transformers-js": patch
"@browser-ai/web-llm": patch
"@browser-ai/core": patch
---

Parse structured argument values in Python-style tool calls.

Argument values were stored as raw text, so a call like
`[ask_user(questions=[{'question': 'How many?'}])]` produced a string
containing Python syntax instead of an array. Values are now parsed as JSON or
Python literals, covering lists, dicts, numbers, and `True`/`False`/`None`.

Calls whose arguments contain parentheses (`q="budget (roughly)?"`) are also no
longer truncated at the first `)`.
