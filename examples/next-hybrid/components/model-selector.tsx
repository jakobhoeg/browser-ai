"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownIcon } from "lucide-react";

const demos = [
  {
    href: "/",
    label: "@browser-ai/core",
  },
  {
    href: "/web-llm",
    label: "@browser-ai/web-llm",
  },
  {
    href: "/transformers-js",
    label: "@browser-ai/transformers-js",
  },
  {
    href: "/wllama",
    label: "@browser-ai/wllama",
  },
];

export function ModelSelector() {
  const pathname = usePathname();
  const currentDemo = demos.find((demo) => demo.href === pathname);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="text-lg font-semibold px-2 max-w-56 sm:max-w-full flex items-center justify-between"
        >
          <span className="truncate mr-2">
            {currentDemo?.label ?? "Select a demo"}
          </span>
          <ChevronDownIcon className="h-4 w-4 flex-shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {demos.map((demo) => (
          <Link href={demo.href} key={demo.href} passHref>
            <DropdownMenuItem>{demo.label}</DropdownMenuItem>
          </Link>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
