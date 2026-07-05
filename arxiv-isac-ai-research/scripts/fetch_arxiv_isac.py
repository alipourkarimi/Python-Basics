#!/usr/bin/env python3
"""Search arXiv for Integrated Sensing and Communication (ISAC) + AI papers.

Queries the official arXiv API (https://info.arxiv.org/help/api/index.html)
for several AI-technique categories (LLMs, generative AI, reinforcement
learning, deep learning) and prints the results as markdown.

Usage:
    python fetch_arxiv_isac.py                 # print to stdout
    python fetch_arxiv_isac.py -o report.md    # write to a file
    python fetch_arxiv_isac.py -n 20           # 20 results per category

Only the standard library is used — no packages to install.
"""

import argparse
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

# Each category: (heading, arXiv fielded search query)
CATEGORIES = [
    (
        "Large Language Models (LLMs)",
        'all:"integrated sensing and communication" AND '
        '(all:"large language model" OR all:LLM)',
    ),
    (
        "Generative AI / Diffusion Models",
        'all:"integrated sensing and communication" AND '
        '(all:"generative AI" OR all:"diffusion model")',
    ),
    (
        "Reinforcement Learning",
        'all:"integrated sensing and communication" AND '
        'all:"reinforcement learning"',
    ),
    (
        "Deep Learning",
        'all:"integrated sensing and communication" AND all:"deep learning"',
    ),
]


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Run one query against the arXiv API and return parsed entries."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "isac-ai-paper-collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        feed = ET.fromstring(response.read())

    papers = []
    for entry in feed.findall(f"{ATOM}entry"):
        authors = [
            author.findtext(f"{ATOM}name", default="")
            for author in entry.findall(f"{ATOM}author")
        ]
        papers.append(
            {
                "id": entry.findtext(f"{ATOM}id", default="").rsplit("/", 1)[-1],
                "title": " ".join(entry.findtext(f"{ATOM}title", default="").split()),
                "authors": authors,
                "published": entry.findtext(f"{ATOM}published", default="")[:10],
                "summary": " ".join(entry.findtext(f"{ATOM}summary", default="").split()),
                "link": entry.findtext(f"{ATOM}id", default=""),
            }
        )
    return papers


def format_markdown(results: dict[str, list[dict]]) -> str:
    lines = [
        "# ISAC + AI papers on arXiv",
        "",
        f"Generated on {time.strftime('%Y-%m-%d')} via the arXiv API.",
    ]
    for heading, papers in results.items():
        lines += ["", f"## {heading}", ""]
        if not papers:
            lines.append("_No results returned._")
        for paper in papers:
            first_authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                first_authors += " et al."
            lines += [
                f"### [{paper['title']}]({paper['link']})",
                "",
                f"- **arXiv:** {paper['id']} · {paper['published']} · {first_authors}",
                "",
                f"> {paper['summary']}",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--max-results", type=int, default=10,
                        help="results per category (default: 10)")
    parser.add_argument("-o", "--output", help="write markdown to this file")
    args = parser.parse_args()

    results = {}
    for index, (heading, query) in enumerate(CATEGORIES):
        print(f"Searching arXiv: {heading} ...")
        results[heading] = search_arxiv(query, args.max_results)
        # arXiv API etiquette: no more than one request every 3 seconds.
        if index < len(CATEGORIES) - 1:
            time.sleep(3)

    markdown = format_markdown(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f"Wrote {args.output}")
    else:
        print()
        print(markdown)


if __name__ == "__main__":
    main()
