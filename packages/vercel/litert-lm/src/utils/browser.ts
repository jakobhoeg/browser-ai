declare global {
  interface Navigator {
    gpu?: GPU;
  }
}

export function isMobile(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  );
}

/**
 * LiteRT-LM (LiteRT.js) runs models in the browser via WebGPU, so WebGPU
 * support is the hard requirement for this provider.
 */
export function checkWebGPU(): boolean {
  try {
    return !!globalThis?.navigator?.gpu;
  } catch {
    return false;
  }
}

/**
 * Check if the browser supports the LiteRT-LM Web runtime.
 */
export function doesBrowserSupportLiteRTLM(): boolean {
  return checkWebGPU();
}
