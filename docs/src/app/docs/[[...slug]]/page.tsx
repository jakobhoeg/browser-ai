import { PageContextMenu } from "@/components/page-context-menu";
import { getPageImage, getLLMText, source } from "@/lib/source";
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from "fumadocs-ui/layouts/docs/page";
import { notFound, redirect } from "next/navigation";
import { getMDXComponents } from "@/mdx-components";
import type { Metadata } from "next";

export default async function Page(props: PageProps<"/docs/[[...slug]]">) {
  const params = await props.params;

  // Redirect /docs to /docs/ai-sdk-v7
  if (!params.slug || params.slug.length === 0) {
    redirect("/docs/ai-sdk-v7");
  }

  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const markdown = await getLLMText(page);

  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <div className="flex items-center justify-between gap-4">
        <DocsTitle className="mb-0">{page.data.title}</DocsTitle>
        <PageContextMenu markdown={markdown} />
      </div>
      <DocsDescription>{page.data.description}</DocsDescription>
      <DocsBody>
        <MDX components={getMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(
  props: PageProps<"/docs/[[...slug]]">,
): Promise<Metadata> {
  const params = await props.params;

  if (!params.slug || params.slug.length === 0) {
    return {};
  }

  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      images: getPageImage(page).url,
    },
  };
}
