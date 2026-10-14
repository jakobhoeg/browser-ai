/**
 * Check if the browser exposes WebGPU.
 */
export function checkWebGPU(): boolean {
  const checkGPU = !!globalThis?.navigator?.gpu;
  if (isFirefox()) {
    // firefox with WebGPU support is very slow, so disabled it in default.
    return false;
  } else {
    return checkGPU;
  }
}

/**
 * Check if the runtime supports WebAssembly JavaScript Promise Integration (JSPI).
 */
export function checkJSPI(): boolean {
  return typeof WebAssembly !== "undefined" && "Suspending" in WebAssembly;
}

/**
 * Check if the browser support (fine for chrome, safari, and firefox so always return true)
 * We should change this logics if there are browsers that cannot run.
 */
export function doesBrowserSupportWllama(): boolean {
  return true;
}

export const isSupportMem64 = (): boolean => {
  try {
    new WebAssembly.Memory({
      address: 'i64',
      initial: 1n, // 1 page (64 KiB)
    } as any);
    return true;
  } catch {
    return false;
  }
};

/**
 * Check if the browser is Firefox
 */
export const isFirefox = (): boolean => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
