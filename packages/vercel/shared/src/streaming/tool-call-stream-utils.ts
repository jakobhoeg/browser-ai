/**
 * Extracts tool name from partial fence content for early tool-input-start emission.
 * Expects a JSON fragment like: {"name":"toolName"
 */
export function extractToolName(content: string): string | null {
  const jsonMatch = content.match(/\{\s*"name"\s*:\s*"([^"]+)"/);
  if (jsonMatch) {
    return jsonMatch[1];
  }
  return null;
}

export interface ArgumentsStreamState {
  searchFrom: number;
  valueStartIndex: number | null;
  parseIndex: number;
  started: boolean;
  depth: number;
  inString: boolean;
  escaped: boolean;
  complete: boolean;
}

const ARGUMENTS_FIELD_REGEX = /"arguments"\s*:\s*/g;
const ARGUMENTS_SEARCH_OVERLAP = 32;

export function createArgumentsStreamState(): ArgumentsStreamState {
  return {
    searchFrom: 0,
    valueStartIndex: null,
    parseIndex: 0,
    started: false,
    depth: 0,
    inString: false,
    escaped: false,
    complete: false,
  };
}

/**
 * Incrementally extracts only new argument content from a streaming tool call fence.
 */
export function extractArgumentsDelta(
  content: string,
  state: ArgumentsStreamState,
): string {
  if (state.complete) {
    return "";
  }

  if (state.valueStartIndex === null) {
    ARGUMENTS_FIELD_REGEX.lastIndex = state.searchFrom;
    const match = ARGUMENTS_FIELD_REGEX.exec(content);
    ARGUMENTS_FIELD_REGEX.lastIndex = 0;

    if (!match || match.index === undefined) {
      state.searchFrom = Math.max(0, content.length - ARGUMENTS_SEARCH_OVERLAP);
      return "";
    }

    state.valueStartIndex = match.index + match[0].length;
    state.parseIndex = state.valueStartIndex;
    state.searchFrom = state.valueStartIndex;
  }

  if (state.parseIndex >= content.length) {
    return "";
  }

  let delta = "";
  for (let i = state.parseIndex; i < content.length; i++) {
    const char = content[i];
    delta += char;

    if (!state.started) {
      if (!/\s/.test(char)) {
        state.started = true;
        if (char === "{" || char === "[") {
          state.depth = 1;
        }
      }
      continue;
    }

    if (state.escaped) {
      state.escaped = false;
      continue;
    }

    if (char === "\\") {
      state.escaped = true;
      continue;
    }

    if (char === '"') {
      state.inString = !state.inString;
      continue;
    }

    if (!state.inString) {
      if (char === "{" || char === "[") {
        state.depth += 1;
      } else if (char === "}" || char === "]") {
        if (state.depth > 0) {
          state.depth -= 1;
          if (state.depth === 0) {
            state.parseIndex = i + 1;
            state.complete = true;
            return delta;
          }
        }
      }
    }
  }

  state.parseIndex = content.length;
  return delta;
}
