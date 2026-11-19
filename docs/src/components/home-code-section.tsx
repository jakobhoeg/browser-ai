"use client";

import { useState } from "react";
import HomeSnippetInstall from "./home-snippet-install";
import { HomeCodeBlock } from "./home-code-block";
import { CornerPlus } from "./ui/corner-plus";

const PACKAGES = [
  { id: "core", label: "@browser-ai/core" },
  { id: "transformers-js", label: "@browser-ai/transformers-js" },
  { id: "web-llm", label: "@browser-ai/web-llm" },
  { id: "wllama", label: "@browser-ai/wllama" },
] as const;

export type PackageId = (typeof PACKAGES)[number]["id"];

export function HomeCodeSection() {
  const [activePackage, setActivePackage] = useState<PackageId>("core");

  return (
    <div className="mt-10 sm:mt-20 relative before:absolute before:top-0 before:h-px before:w-[200vw] before:bg-gray-950/5 dark:before:bg-white/10 before:-left-[100vw] after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(300px,2fr)_minmax(0,3fr)]">
        {/* Install snippet section */}
        <div className="hidden sm:flex border-(--pattern-fg) bg-[image:repeating-linear-gradient(315deg,_var(--pattern-fg)_0,_var(--pattern-fg)_1px,_transparent_0,_transparent_50%)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-black)]/5 dark:[--pattern-fg:var(--color-white)]/10 items-center justify-center lg:px-8 py-8 lg:py-12 lg:border-r max-lg:border-b">
          <HomeSnippetInstall
            value={activePackage}
            onValueChange={setActivePackage}
          />
        </div>

        {/* Code block section */}
        <div className="relative bg-gray-950/5 p-2 dark:bg-white/10">
          <CornerPlus />

          <HomeCodeBlock
            value={activePackage}
            onValueChange={setActivePackage}
          />
        </div>
      </div>
    </div>
  );
}
