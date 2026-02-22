"use client";

import { useRef, type ReactNode } from "react";
import { CopyIcon, DownloadIcon } from "lucide-react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { BrowserAILogo } from "./browser-ai-logo";
import type { SVGProps } from "react";

export function BrowserAILogoWithMenu({
  children,
  ...props
}: SVGProps<SVGSVGElement> & { children?: ReactNode }) {
  const svgRef = useRef<SVGSVGElement>(null);

  const getSvgString = () => {
    if (!svgRef.current) return "";
    const clone = svgRef.current.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("width", "300");
    clone.setAttribute("height", "300");
    return `<?xml version="1.0" encoding="UTF-8"?>\n${clone.outerHTML}`;
  };

  const copyAsSvg = async () => {
    await navigator.clipboard.writeText(getSvgString());
  };

  const downloadAsSvg = () => {
    const svg = getSvgString();
    const blob = new Blob([svg], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "browser-ai-logo.svg";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <ContextMenu>
      <ContextMenuTrigger className="flex items-center gap-1">
        <BrowserAILogo ref={svgRef} {...props} />
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={copyAsSvg}>
          <CopyIcon className="mr-1 size-4" />
          Copy logo as SVG
        </ContextMenuItem>
        <ContextMenuItem onClick={downloadAsSvg}>
          <DownloadIcon className="mr-1 size-4" />
          Download logo as SVG
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
