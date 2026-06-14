import { Suspense } from "react";
import Image from "next/image";
import Link from "next/link";
import { getTweet } from "react-tweet/api";
import { enrichTweet } from "react-tweet";

async function TweetContent({ id }: { id: string }) {
  const tweet = await getTweet(id);
  if (!tweet) return null;

  const { user, entities, url } = enrichTweet(tweet);

  return (
    <Link
      href={url}
      target="_blank"
      rel="noopener"
      className="flex flex-col gap-4 border border-gray-950/10 bg-secondary dark:bg-background p-4 transition-colors hover:bg-card dark:border-white/10 dark:hover:bg-card"
    >
      <div className="flex items-center gap-3">
        <Image
          src={user.profile_image_url_https}
          alt={user.name}
          width={40}
          height={40}
          className="size-10 rounded-full"
          unoptimized
        />
        <div className="flex flex-col">
          <span className="text-sm font-medium">{user.name}</span>
          <span className="text-xs text-black/40 dark:text-white/40">
            @{user.screen_name}
          </span>
        </div>
      </div>
      <p className="text-sm leading-relaxed text-black/70 dark:text-white/70">
        {entities.map((item, i) => {
          switch (item.type) {
            case "hashtag":
            case "mention":
            case "url":
            case "symbol":
              return (
                <span key={i} className="text-blue-500">
                  {item.text}
                </span>
              );
            case "media":
              return null;
            default:
              return <span key={i}>{item.text}</span>;
          }
        })}
      </p>
    </Link>
  );
}

function TweetSkeleton() {
  return (
    <div className="flex flex-col gap-4 border h-44 border-gray-950/10 bg-background p-4 dark:border-white/10 animate-pulse">
      <div className="flex items-center gap-3">
        <div className="size-10 rounded-full bg-gray-200 dark:bg-white/10" />
        <div className="flex flex-col gap-1">
          <div className="h-4 w-24 rounded bg-gray-200 dark:bg-white/10" />
          <div className="h-3 w-16 rounded bg-gray-200 dark:bg-white/10" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-4 w-full rounded bg-gray-200 dark:bg-white/10" />
        <div className="h-4 w-3/4 rounded bg-gray-200 dark:bg-white/10" />
      </div>
    </div>
  );
}

export function TweetCard({ id }: { id: string }) {
  return (
    <Suspense fallback={<TweetSkeleton />}>
      <TweetContent id={id} />
    </Suspense>
  );
}
