#!/usr/bin/env python3
"""
Extract main article content for URLs listed in a CSV and save to JSONL.

This version is tailored for:
switchgear_construction_results2124.csv
Columns: Year, Keyword, Result Title, Link, Extracted Title, Summary

Each row is a unique webpage.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Optional extractors
try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from readability import Document
except Exception:
    Document = None

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Column mapping for your CSV
COL_LINK = "Link"
COL_TITLE = "Result Title"
COL_YEAR = "Year"
COL_KEYWORD = "Keyword"
COL_PROVIDED_EXTRACTED_TITLE = "Extracted Title"
COL_SUMMARY = "Summary"


def make_session(total_retries=3, backoff_factor=0.5, timeout=20) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    session.request_timeout = timeout
    return session


def fetch_html(session, url):
    try:
        resp = session.get(url, timeout=getattr(session, "request_timeout", 20), allow_redirects=True)
        ct = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "text/html" in ct.lower():
            return resp.text, resp.status_code, None
        else:
            return None, resp.status_code, f"Unexpected content-type/status: {ct} / {resp.status_code}"
    except requests.RequestException as e:
        return None, None, str(e)


def extract_with_trafilatura(html, url):
    if trafilatura is None:
        return None, None, {}
    try:
        downloaded = html
        config = trafilatura.settings.use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        text = trafilatura.extract(
            downloaded, url=url, include_formatting=False,
            include_tables=False, with_metadata=True, config=config
        )
        meta = trafilatura.extract_metadata(downloaded, url=url) or {}
        if text is None:
            text_only = trafilatura.extract(
                downloaded, url=url, include_formatting=False, include_tables=False, config=config
            )
            return text_only, None, dict(meta) if hasattr(meta, "as_dict") else meta if isinstance(meta, dict) else {}
        return text, None, dict(meta) if hasattr(meta, "as_dict") else meta if isinstance(meta, dict) else {}
    except Exception:
        return None, None, {}


def extract_with_readability(html, url):
    if Document is None:
        return None, None
    try:
        doc = Document(html)
        title = doc.short_title()
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text("\n", strip=True)
        return text, title
    except Exception:
        return None, None


def extract_fallback_text(html):
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    body = soup.body or soup
    return body.get_text("\n", strip=True)


@dataclass
class ExtractionResult:
    href: str
    title: Optional[str] = None
    source: Optional[str] = None
    published: Optional[str] = None
    snippet: Optional[str] = None
    query: Optional[str] = None

    http_status: Optional[int] = None
    fetch_error: Optional[str] = None

    extracted_title: Optional[str] = None
    extracted_text: Optional[str] = None
    extractor: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_main_content(html, url):
    text, _, meta = extract_with_trafilatura(html, url)
    if text:
        return text, None, "trafilatura", meta or {}

    text2, title2 = extract_with_readability(html, url)
    if text2:
        return text2, title2, "readability", {}

    text3 = extract_fallback_text(html)
    return text3, None, "bs4-fallback", {}


def process_row(row, session):
    href = row.get(COL_LINK)
    if not href:
        raise ValueError(f"Row missing URL in column '{COL_LINK}'")

    result = ExtractionResult(
        href=href,
        title=row.get(COL_TITLE),
        source=None,
        published=str(row.get(COL_YEAR)) if pd.notna(row.get(COL_YEAR)) else None,
        snippet=row.get(COL_SUMMARY),
        query=row.get(COL_KEYWORD),
    )

    extra_meta = {
        "csv_year": row.get(COL_YEAR),
        "csv_keyword": row.get(COL_KEYWORD),
        "csv_result_title": row.get(COL_TITLE),
        "csv_provided_extracted_title": row.get(COL_PROVIDED_EXTRACTED_TITLE),
        "csv_summary": row.get(COL_SUMMARY),
    }

    html, status, error = fetch_html(session, href)
    result.http_status = status
    if error:
        result.fetch_error = error

    if html:
        text, extracted_title, extractor_name, meta = extract_main_content(html, href)
        result.extracted_text = text
        result.extracted_title = extracted_title
        result.extractor = extractor_name
        merged_meta = {}
        if isinstance(meta, dict):
            merged_meta.update(meta)
        merged_meta.update({k: v for k, v in extra_meta.items() if v is not None})
        result.metadata = merged_meta

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="switchgear_construction_results2425.csv",
                        help="Path to input CSV")
    parser.add_argument("--output", "-o", default=None, help="Path to output JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows")
    parser.add_argument("--start", type=int, default=0, help="Optional start index")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")
    args = parser.parse_args()

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_extracted.jsonl"

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input, encoding=args.encoding)
    required = {COL_LINK, COL_TITLE}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    if args.start or args.limit:
        end = None if args.limit is None else args.start + args.limit
        df = df.iloc[args.start:end]

    session = make_session()
    total = len(df)
    print(f"Processing {total} rows...")

    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            try:
                result = process_row(row.to_dict(), session)
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                error_record = {
                    "href": row.get(COL_LINK),
                    "title": row.get(COL_TITLE),
                    "error": str(e),
                    "phase": "process_row",
                }
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

            if written and written % 10 == 0:
                print(f"Wrote {written}/{total}")

    print(f"Done. Wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()
