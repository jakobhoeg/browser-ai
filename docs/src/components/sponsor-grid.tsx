import { cn } from "@/lib/utils";
import Link from "next/link";

type Sponsor = {
  name: string;
  href: string;
  logo: React.ComponentType<{ className?: string }>;
};

export default function SponsorGrid({ sponsors }: { sponsors: Sponsor[] }) {
  return (
    <div className="relative after:absolute after:bottom-0 after:h-px after:w-[200vw] after:bg-gray-950/5 dark:after:bg-white/10 after:-left-[100vw]">
      <ul className="flex flex-col sm:flex-row sm:flex-wrap sm:justify-center gap-2 sm:gap-5 divide-y divide-gray-950/5 dark:divide-white/10 sm:divide-y-0 bg-[image:repeating-linear-gradient(315deg,_var(--pattern-fg)_0,_var(--pattern-fg)_1px,_transparent_0,_transparent_50%)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-black)]/5 dark:[--pattern-fg:var(--color-white)]/10">
        {sponsors.map((sponsor, index) => (
          <li
            key={sponsor.href}
            className={cn(
              "flex sm:border-x sm:border-separate sm:border-gray-950/5 sm:dark:border-white/10",
            )}
          >
            <Link
              href={sponsor.href}
              target="_blank"
              rel="noopener sponsored"
              className="flex w-full items-center justify-center gap-3 px-6 py-6 sm:px-8 sm:py-8 transition-colors bg-background hover:bg-gray-100 dark:hover:bg-zinc-900"
            >
              <sponsor.logo className="size-8 shrink-0" />
              <p className="text-lg ">{sponsor.name}</p>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
