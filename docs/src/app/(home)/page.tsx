import { ChromeLogo } from "@/components/logos";
import { HomeCodeSection } from "@/components/home-code-section";
import SponsorGrid from "@/components/sponsor-grid";
import { TweetCard } from "@/components/tweet-card";
import { TweetGrid } from "@/components/tweet-grid";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MoveUpRight } from "lucide-react";
import Link from "next/link";
import { VercelOSSLogo } from "@/components/logos/vercel-oss-logo";

const tweetIds = [
  "1995030252259774593",
  "2012275639865217189",
  "2008848683857830005",
  "1980659099445653624",
  "1981060426315288776",
  "1957752385272738170",
  "1980646442441285656",
  "1957771454839238912",
  "1957762133787082766",
];

const sponsors = [
  {
    name: "Chrome for Developers",
    href: "https://developer.chrome.com/",
    logo: ChromeLogo,
  },
  {
    name: "",
    href: "https://vercel.com/oss",
    logo: VercelOSSLogo,
  },
];

export default function HomePage() {
  return (
    <div className="grid min-h-dvh grid-cols-1 grid-rows-[1fr_1px_auto_1px_auto] justify-center [--gutter-width:2.5rem] md:-mx-4 md:grid-cols-[var(--gutter-width)_minmax(0,var(--breakpoint-xl))_var(--gutter-width)] lg:mx-0">
      {/* Left vertical separators */}
      <VerticalSeparatorLeft />

      {/* Main content */}
      <main className="grid gap-24 pb-24 text-gray-950 sm:gap-40 md:pb-40 dark:text-white">
        <div>
          <div className="relative flex h-16 items-end px-2 font-mono tracking-tighter text-xs/6 whitespace-pre text-black/40 max-sm:px-4 sm:h-24 dark:text-white/40 after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
            Model providers for Vercel AI SDK v5 & v6
          </div>

          <div className="relative before:absolute before:top-0 before:h-px before:w-[200vw] before:-left-[100vw] after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
            <h1 className="px-2 text-4xl tracking-tighter text-balance max-lg:font-medium max-sm:px-4 sm:text-5xl lg:text-6xl xl:text-8xl">
              Build local, in-browser AI applications with ease.
            </h1>
          </div>

          <div className="mt-5 font-mono tracking-tighter relative text-black/40 dark:text-white/40 px-2 max-sm:px-4 before:absolute before:top-0 before:h-px before:w-[200vw] before:-left-[100vw] after:absolute after:bottom-0 after:h-px">
            Framework agnostic. Built-in state management. Tool calling.
            Structured output. Streaming.
          </div>

          <Separator />

          <div className="mt-10 flex gap-2 px-2 max-sm:px-4 relative before:absolute before:top-0 before:h-px before:w-[200vw] before:bg-gray-950/5 dark:before:bg-white/10 before:-left-[100vw] after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
            <Button asChild className="tracking-tight text-balance">
              <Link href="/docs">Get started building</Link>
            </Button>

            <Button
              variant="secondary"
              className="tracking-tight text-balance flex gap-2"
            >
              <Link
                href="https://ai-sdk.dev/docs/introduction"
                target="_blank"
                rel="noopener sponsored"
              >
                Vercel AI SDK
              </Link>
              <MoveUpRight className="size-3.5" />
            </Button>
          </div>

          <HomeCodeSection />
          <div className="relative mt-20 max-w-full">
            <div className="relative before:absolute before:top-0 before:h-px before:w-[200vw] before:bg-gray-950/5 dark:before:bg-white/10 before:h-px before:w-[200vw] before:bg-gray-950/5 dark:before:bg-white/10 before:-left-[100vw] after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
              <h2 className="max-w-2xl px-2 text-4xl font-medium tracking-tighter text-balance max-sm:px-4">
                Backed by the creators
              </h2>
            </div>
            <div className="relative items-center px-2 font-mono text-xs/6 text-black/40 dark:text-white/40 max-sm:px-4 flex after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
              <p className="text-balance">
                This project is proudly sponsored by the creators behind the{" "}
                <a
                  href="https://developer.chrome.com/docs/ai/built-in"
                  target="_blank"
                  rel="noopener sponsored"
                  className="underline"
                >
                  Built-in AI
                </a>{" "}
                initiative, and is part of the{" "}
                <a
                  href="https://vercel.com/oss"
                  target="_blank"
                  rel="noopener"
                  className="underline"
                >
                  Vercel OSS Program
                </a>
                .
              </p>
            </div>

            <div className="relative h-10 items-end px-2 font-mono text-xs/6 whitespace-pre text-black/40 dark:text-white/40 max-sm:px-4 flex after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]"></div>
            <SponsorGrid sponsors={sponsors} />
          </div>
          {/* Testimonials */}
          <div className="relative mt-20 max-w-full">
            <div className="h-4 items-end px-2 font-mono text-xs/6 whitespace-pre text-black/40 dark:text-white/40 max-sm:px-4 flex">
              Testimonials
            </div>
            <div className="relative before:absolute before:top-0 before:h-px before:w-[200vw] before:bg-gray-950/5 dark:before:bg-white/10 before:-left-[100vw] after:absolute after:bottom-0 after:h-px">
              <h2 className="max-w-2xl px-2 text-4xl font-medium tracking-tighter text-balance max-sm:px-4">
                What developers are saying
              </h2>
            </div>
            <TweetGrid totalCount={tweetIds.length}>
              {tweetIds.map((id) => (
                <TweetCard key={id} id={id} />
              ))}
            </TweetGrid>
          </div>
        </div>
      </main>

      {/* Right vertical separators */}
      <VerticalSeparatorRight />
    </div>
  );
}

function Separator({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative flex h-7 lg:h-10 w-full border-y border-edge",
        "bg-[image:repeating-linear-gradient(315deg,_var(--pattern-fg)_0,_var(--pattern-fg)_1px,_transparent_0,_transparent_50%)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-black)]/5 dark:[--pattern-fg:var(--color-white)]/10",
        "before:absolute before:-z-1 before:h-7 lg:before:h-10 before:w-[100vw] before:right-[calc(100%+var(--gutter-width))] before:border-y before:border-edge",
        "after:absolute after:-z-1 after:h-7 lg:after:h-10 after:w-[100vw] after:left-[calc(100%+var(--gutter-width))] after:border-y after:border-edge",
        className,
      )}
    />
  );
}

function VerticalSeparatorRight({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "row-span-full row-start-1 hidden border-x border-x-(--pattern-fg) bg-[image:repeating-linear-gradient(315deg,_var(--pattern-fg)_0,_var(--pattern-fg)_1px,_transparent_0,_transparent_50%)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-black)]/7 md:col-start-3 md:block dark:[--pattern-fg:var(--color-white)]/8",
        className,
      )}
    />
  );
}

function VerticalSeparatorLeft({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "col-start-1 row-span-full row-start-1 hidden border-x border-x-(--pattern-fg) bg-[image:repeating-linear-gradient(315deg,_var(--pattern-fg)_0,_var(--pattern-fg)_1px,_transparent_0,_transparent_50%)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-black)]/7 md:block dark:[--pattern-fg:var(--color-white)]/8",
        className,
      )}
    />
  );
}
