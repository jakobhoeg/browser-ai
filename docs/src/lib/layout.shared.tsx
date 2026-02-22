import { GithubInfo } from "fumadocs-ui/components/github-info";
import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { BrowserAILogoWithMenu } from "@/components/logos/browser-ai-logo-with-menu";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <BrowserAILogoWithMenu className="dark:invert" width={34} height={24}>
          <p className="hidden sm:flex text-sm text-muted-foreground">
            @browser-ai
          </p>
        </BrowserAILogoWithMenu>
      ),
    },
    githubUrl: "https://github.com/jakobhoeg/browser-ai",
    // links: [
    //   {
    //     type: 'custom',
    //     children: (
    //       <GithubInfo owner="jakobhoeg" repo="browser-ai" />
    //     ),
    //   },
    // ],
  };
}
