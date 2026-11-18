import { HomeLayout } from "fumadocs-ui/layouts/home";
import { baseOptions } from "@/lib/layout.shared";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Browser AI model providers",
  description:
    "Build local, in-browser AI applications with better DX. AI SDK model providers providing built-in state management, tool calling, structured output, streaming and more.",
  keywords: [
    "AI SDK",
    "Vercel AI",
    "browser-ai",
    "Browser AI",
    "Built-in AI",
    "browser AI",
    "local AI",
    "Chrome AI",
    "WebLLM",
    "Transformers.js",
    "wllama",
    "llama.cpp",
    "GGUF",
    "language model",
    "machine learning",
  ],
  authors: [{ name: "Jakob Hoeg", url: "https://jakobhoeg.dev" }],
  creator: "Jakob Hoeg",
  metadataBase: new URL("https://browser-ai.dev"),
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://browser-ai.dev",
    title: "Browser AI model providers",
    description: "Build local, in-browser AI applications with better DX.",
    siteName: "Browser AI",
    images: [
      {
        url: "/opengraph-image.jpeg",
        width: 1280,
        height: 720,
        alt: "Browser AI model providers",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Browser AI model providers",
    description: "Build local, in-browser AI applications with better DX.",
    creator: "@hoeg_jakob",
    images: ["/opengraph-image.jpeg"],
  },
};

export default function Layout({ children }: LayoutProps<"/">) {
  return (
    <HomeLayout className="max-w-screen overflow-x-hidden" {...baseOptions()}>
      {children}
    </HomeLayout>
  );
}
