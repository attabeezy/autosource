"""
AutoSource: Report Fetcher
Query World Bank Documents & Reports API → Download PDFs → Catalog

Usage:
    uv run fetch_reports.py --topic "credit" --limit 10
    uv run fetch_reports.py --region Africa --topic "poverty" --limit 20
"""

import argparse
import json
import time
from pathlib import Path

import requests


SOURCE_DIR = Path("source_library")
CATALOG_FILE = SOURCE_DIR / "catalog.json"

DOCS_API_BASE = "https://search.worldbank.org/api/v3/wds"


def search_docs_api(
    region: str = "AFR",
    topic: str = "",
    year: int = 2020,
    limit: int = 10,
) -> list[dict]:
    """Search World Bank Documents & Reports API."""
    params = {
        "format": "json",
        "strdate": f"{year}-01-01",
        "rows": limit,
        "fl": "pdfurl,display_title,docdt,topic,count,repnme,abstracts,guid",
    }

    search_term = region if region else ""
    if topic:
        search_term = f"{search_term} {topic}".strip() if search_term else topic

    if search_term:
        params["qterm"] = search_term

    print(f"Querying World Bank Documents API...")
    print(f"  Search: '{search_term or '(all)'}'")
    print(f"  Year: >={year}")

    response = requests.get(DOCS_API_BASE, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    docs_data = data.get("documents", {})
    total = data.get("total", 0)

    results = docs_data.get("document", [])
    if not results:
        if isinstance(docs_data, dict):
            doc_keys = [k for k in docs_data.keys() if k != "facets"]
            results = [docs_data.get(k) for k in doc_keys] if doc_keys else []
    elif not isinstance(results, list):
        results = [results]
    elif isinstance(results, dict):
        doc_keys = [k for k in results.keys() if k != "facets"]
        results = [results[k] for k in doc_keys] if doc_keys else []

    print(f"  Found {len(results)} reports (total matches: {total})")

    reports = []
    for r in results:
        pdf_url = r.get("pdfurl", "")
        if not pdf_url:
            continue

        reports.append(
            {
                "source": "docs",
                "guid": r.get("guid", ""),
                "title": r.get("display_title", r.get("repnme", "Untitled")),
                "date": r.get("docdt", ""),
                "countries": r.get("count", ""),
                "topics": r.get("topic", ""),
                "abstract": (
                    r.get("abstracts", {}).get("cdata!") or r.get("abstracts", "") or ""
                )[:500],
                "url": pdf_url,
                "filename": None,
                "selected": False,
            }
        )

    return reports


def download_file(url: str, output_path: Path) -> bool:
    """Download file to output_path."""
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def save_catalog(reports: list[dict], catalog_path: Path) -> None:
    """Save catalog JSON."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(
            {"reports": reports, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            f,
            indent=2,
        )
    print(f"Catalog saved to {catalog_path}")


def generate_selector(reports: list[dict], output_path: Path) -> None:
    """Generate selector.md for manual selection."""
    lines = [
        "# Report Selection",
        "",
        "Review the reports below. Toggle `[x]` to select for processing.",
        "",
        "## Available Reports",
        "",
    ]

    source_icons = {"docs": "[PDF]"}
    docs_reports = [r for r in reports if r.get("source") == "docs"]

    if docs_reports:
        lines.append("### World Bank Documents & Reports")
        lines.append("")
        for i, r in enumerate(docs_reports, 1):
            selected = r.get("selected", False)
            checkbox = "[x]" if selected else "[ ]"
            title = r.get("title", "Untitled")[:80]
            date = r.get("date", "unknown")
            countries = r.get("countries", "multiple")
            filename = r.get("filename", "N/A")
            lines.append(f"- [{checkbox}] **{i}. {title}**")
            lines.append(f"  - Date: {date} | Countries: {countries}")
            if filename and filename != "N/A":
                lines.append(f"  - File: `{filename}`")
            lines.append("")

    lines.extend(
        [
            "## Processing Selected Reports",
            "",
            "After selecting, run:",
            "```bash",
            "uv run extract.py --pdf source_library/<selected_file>.pdf",
            "```",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"Selector saved to {output_path}")


def slug_from_title(title: str, index: int) -> str:
    """Create filename slug from title."""
    return (
        title[:40].lower().replace(" ", "_").replace("/", "-").replace(":", "")[:40]
        + f"_{index:02d}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch World Bank reports")
    parser.add_argument("--topic", default="", help="Search topic/keyword")
    parser.add_argument("--region", default="AFR", help="Region code (default: AFR)")
    parser.add_argument("--year", type=int, default=2020, help="Minimum year")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument(
        "--output-dir", default=str(SOURCE_DIR), help="Output directory"
    )
    parser.add_argument(
        "--download/--no-download",
        default=True,
        dest="download",
        help="Download PDFs",
    )
    parser.add_argument(
        "--auto-select/--no-auto-select",
        default=False,
        dest="auto_select",
        help="Auto-select all for processing",
    )
    args = parser.parse_args()

    source_dir = Path(args.output_dir)

    reports = search_docs_api(
        region=args.region,
        topic=args.topic,
        year=args.year,
        limit=args.limit,
    )

    if not reports:
        print("No reports found")
        return

    downloaded = []
    for i, r in enumerate(reports):
        title_slug = slug_from_title(r.get("title", "report"), i + 1)
        source = r.get("source", "docs")

        if source == "docs":
            ext = ".pdf"
            filename = f"{title_slug}{ext}"
        else:
            ext = ".html"
            filename = f"{title_slug}{ext}"

        filepath = source_dir / filename

        if args.download and r.get("url"):
            print(f"\nDownloading {i + 1}/{len(reports)}: {r['title'][:50]}...")

            if source == "docs":
                if download_file(r["url"], filepath):
                    r["filename"] = filename
                    r["selected"] = True
                    downloaded.append(filename)
                else:
                    r["filename"] = None
                    r["selected"] = args.auto_select
            else:
                r["filename"] = filename
                r["selected"] = args.auto_select
                downloaded.append(filename)
        else:
            r["filename"] = filename if filepath.exists() else None
            r["selected"] = args.auto_select

    save_catalog(reports, CATALOG_FILE)

    selector_path = source_dir / "selector.md"
    generate_selector(reports, selector_path)

    print(f"\n=== Summary ===")
    print(f"Reports found: {len(reports)}")
    print(f"Files downloaded: {len(downloaded)}")
    print(f"Catalog: {CATALOG_FILE}")
    print(f"Selector: {selector_path}")

    if downloaded:
        print(f"\nTo process a file:")
        print(f"  uv run extract.py --pdf source_library/{downloaded[0]}")


if __name__ == "__main__":
    main()
