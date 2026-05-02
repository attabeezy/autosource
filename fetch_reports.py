"""
AutoSource: Report Fetcher
Query World Bank Documents & Reports API + NADA Microdata API → Download PDFs → Catalog

Supports two sources:
1. World Bank Documents & Reports (search.worldbank.org/api/v3/wds)
2. World Bank NADA Microdata (microdata.worldbank.org/api/v5)

Usage:
    uv run fetch_reports.py --source both --topic "credit" --limit 10
    uv run fetch_reports.py --source docs --topic "poverty" --limit 20
    uv run fetch_reports.py --source nada --country "Nigeria"
"""

import argparse
import json
import time
from pathlib import Path

import requests


SOURCE_DIR = Path("source_library")
CATALOG_FILE = SOURCE_DIR / "catalog.json"

DOCS_API_BASE = "https://search.worldbank.org/api/v3/wds"
NADA_API_BASE = "https://microdata.worldbank.org/api/v5"


def search_docs_api(
    region: str = "AFR",
    topic: str = "",
    year: int = 2020,
    limit: int = 10,
) -> list[dict]:
    """Search World Bank Documents & Reports API."""
    params = {
        "format": "json",
        "geo_reg": region,
        "docty_exact": "Report",
        "strdate": f"{year}-01-01",
        "rows": limit,
        "fl": "pdfurl,display_title,docdt,topic,count,repnme,abstracts,guid",
    }

    if topic:
        params["qterm"] = topic

    print(f"Querying World Bank Documents API...")
    print(f"  Region: {region}")
    print(f"  Topic: {topic or '(general)'}")
    print(f"  Year: >={year}")

    response = requests.get(DOCS_API_BASE, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = data.get("documents", {}).get("document", [])
    if not isinstance(results, list):
        results = [results] if results else []

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
                "abstract": r.get("abstracts", "")[:500] if r.get("abstracts") else "",
                "url": pdf_url,
                "filename": None,
                "selected": False,
            }
        )

    return reports


def search_nada_api(
    country: str = "",
    topic: str = "",
    year: int = 2020,
    limit: int = 10,
) -> list[dict]:
    """Search World Bank NADA Microdata API."""
    params = {
        "format": "json",
        "rows": limit,
        "sort": "date_desc",
    }

    if country:
        params["country"] = country
    if topic:
        params["search"] = topic

    print(f"Querying World Bank NADA Microdata API...")
    print(f"  Country: {country or '(all)'}")
    print(f"  Topic: {topic or '(all)'}")
    print(f"  Year: >={year}")

    try:
        response = requests.get(
            f"{NADA_API_BASE}/catalog/search", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  NADA API error: {e}")
        return []

    results = data.get("documents", [])[:limit]

    reports = []
    for r in results:
        study_id = r.get("id", "")
        title = r.get("title", r.get("name", "Untitled"))
        year_val = r.get("year", 0)

        if year_val and int(year_val) < year:
            continue

        reports.append(
            {
                "source": "nada",
                "guid": study_id,
                "title": title,
                "date": str(year_val) if year_val else "",
                "countries": r.get("country", ""),
                "topics": r.get("topics", ""),
                "abstract": r.get("description", "")[:500]
                if r.get("description")
                else "",
                "url": f"https://microdata.worldbank.org/study/{study_id}",
                "filename": None,
                "selected": False,
            }
        )

    return reports


def search_both_apis(
    region: str = "AFR",
    topic: str = "",
    year: int = 2020,
    limit: int = 10,
) -> list[dict]:
    """Search both APIs and combine results."""
    docs_reports = search_docs_api(region=region, topic=topic, year=year, limit=limit)
    nada_reports = search_nada_api(
        country=region if region == "AFR" else "", topic=topic, year=year, limit=limit
    )

    all_reports = docs_reports + nada_reports

    for i, r in enumerate(all_reports):
        r["guid"] = f"{r['source']}_{r['guid']}" if r["guid"] else f"{r['source']}_{i}"

    return all_reports


def download_file(url: str, output_path: Path, source: str = "docs") -> bool:
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

    source_icons = {"docs": "[PDF]", "[NADA]": "[NADA]"}
    docs_reports = [r for r in reports if r.get("source") == "docs"]
    nada_reports = [r for r in reports if r.get("source") == "nada"]

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

    if nada_reports:
        lines.append("### World Bank NADA Microdata")
        lines.append("")
        for i, r in enumerate(nada_reports, len(docs_reports) + 1):
            selected = r.get("selected", False)
            checkbox = "[x]" if selected else "[ ]"
            title = r.get("title", "Untitled")[:80]
            date = r.get("date", "unknown")
            countries = r.get("countries", "multiple")
            lines.append(f"- [{checkbox}] **{i}. {title}**")
            lines.append(f"  - Date: {date} | Countries: {countries}")
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
    parser.add_argument(
        "--source",
        default="both",
        choices=["docs", "nada", "both"],
        help="API source: docs (Documents), nada (Microdata), or both",
    )
    parser.add_argument("--region", default="AFR", help="Region code (default: AFR)")
    parser.add_argument("--topic", default="", help="Search topic/keyword")
    parser.add_argument("--country", default="", help="Country name (for NADA)")
    parser.add_argument("--year", type=int, default=2020, help="Minimum year")
    parser.add_argument("--limit", type=int, default=10, help="Max results per source")
    parser.add_argument(
        "--output-dir", default=str(SOURCE_DIR), help="Output directory"
    )
    parser.add_argument(
        "--download/--no-download",
        default=True,
        dest="download",
        help="Download files",
    )
    parser.add_argument(
        "--auto-select/--no-auto-select",
        default=False,
        dest="auto_select",
        help="Auto-select all for processing",
    )
    args = parser.parse_args()

    source_dir = Path(args.output_dir)

    if args.source == "docs":
        reports = search_docs_api(
            region=args.region,
            topic=args.topic,
            year=args.year,
            limit=args.limit,
        )
    elif args.source == "nada":
        reports = search_nada_api(
            country=args.country or args.region,
            topic=args.topic,
            year=args.year,
            limit=args.limit,
        )
    else:
        reports = search_both_apis(
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
                if download_file(r["url"], filepath, source):
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

    docs_count = len([r for r in reports if r.get("source") == "docs"])
    nada_count = len([r for r in reports if r.get("source") == "nada"])
    print(f"Sources: {docs_count} Documents, {nada_count} NADA")

    if downloaded:
        print(f"\nTo process a file:")
        print(f"  uv run extract.py --pdf source_library/{downloaded[0]}")


if __name__ == "__main__":
    main()
