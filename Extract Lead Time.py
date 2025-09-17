"""
Switchgear Lead Time Range Summary (Min - Avg - Max) by Year, with URL counts
Resumable + Offline + LLM + Regex Fallback

Reads URLs from a CSV (default: switchgear_summaries.csv with a `url` column),
downloads page text, and extracts lead-time mentions for electrical switchgear
in construction. Uses OpenAI gpt-4o-mini when online, with a regex fallback.

Final output: a single Excel sheet `lead_time_ranges_by_year` with columns:
  year, urls_count, urls_with_mentions,
  min_low_weeks, min_high_weeks, urls_at_min,
  avg_low_weeks, avg_high_weeks, urls_used_for_avg,
  max_low_weeks, max_high_weeks, urls_at_max

Years covered: 2021, 2022, 2023, 2024 only. Units are weeks.

Heuristic year mapping to minimize "unspecified":
  1) article_date metadata year (if 2021–2024)
  2) year embedded in URL path (/2021/, /2022/, /2023/, /2024/)
  3) most-frequent mention-level year extracted by LLM or regex (if 2021–2024)
Unmapped URLs are excluded from the year buckets, but a console log shows how many.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import sys
import time
import warnings
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import dateparser
from dotenv import load_dotenv
from tqdm import tqdm

# Optional content extractors for cleaner main-text
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

try:
    from readability import Document  # type: ignore
except Exception:
    Document = None

# OpenAI client (SDK >= 1.0) - only needed if not --offline
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Silence noisy NumPy warnings from groupby stats on empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

DEFAULT_MODEL = "gpt-4o-mini"
MONTH_TO_WEEKS = 4.33
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TARGET_YEARS = [2021, 2022, 2023, 2024]

META_DATE_KEYS = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "pubdate"}),
    ("meta", {"name": "publishdate"}),
    ("meta", {"name": "timestamp"}),
    ("meta", {"name": "date"}),
    ("meta", {"itemprop": "datePublished"}),
    ("meta", {"name": "DC.date.issued"}),
    ("meta", {"name": "parsely-pub-date"}),
]

# -------- Regex fallback patterns --------
LT_PAT_1 = re.compile(
    r"\blead\s*-?\s*time(?:s)?\s*(?:is|are|of|for|at|running|run|:)?\s*"
    r"(?:up to\s*)?"
    r"(?P<val1>\d{1,3})\s*"
    r"(?:to|[-–—]|through|thru)?\s*"
    r"(?P<val2>\d{1,3})?\s*"
    r"(?P<unit>weeks?|wks?|months?|mos?|mo)\b",
    re.IGNORECASE,
)
LT_PAT_2 = re.compile(
    r"\b(?P<val1>\d{1,3})\s*(?:to|[-–—])?\s*(?P<val2>\d{1,3})?\s*"
    r"(?P<unit>weeks?|wks?|months?|mos?|mo)\s+lead\s*-?\s*time(?:s)?\b",
    re.IGNORECASE,
)
LT_PAT_3 = re.compile(
    r"\blead\s*-?\s*time(?:s)?\s*(?:exceed(?:ing|s)|over|more than|>+)\s*"
    r"(?P<val1>\d{1,3})\s*(?P<unit>weeks?|wks?|months?|mos?|mo)\b",
    re.IGNORECASE,
)
LT_PAT_4 = re.compile(
    r"\bup to\s*(?P<val1>\d{1,3})\s*(?P<unit>weeks?|wks?|months?|mos?|mo)\s+"
    r"lead\s*-?\s*time(?:s)?\b",
    re.IGNORECASE,
)
YEAR_NEARBY = re.compile(r"\b(2021|2022|2023|2024|2025)\b")

# ------------------------- Utilities -------------------------
def read_urls_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    col = None
    lower_cols = {c.lower(): c for c in df.columns}
    if "url" in lower_cols:
        col = lower_cols["url"]
    else:
        for c in df.columns:
            if df[c].astype(str).str.startswith(("http://", "https://")).any():
                col = c
                break
    if not col:
        raise ValueError("CSV must contain a URL column (named 'url' preferred).")
    urls = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.startswith(("http://", "https://"))]
        .tolist()
    )
    return urls

def fetch_html(url: str, timeout: int = 25) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None

def extract_meta_date(html: str) -> Optional[pd.Timestamp]:
    soup = BeautifulSoup(html, "lxml")
    for tag, attrs in META_DATE_KEYS:
        el = soup.find(tag, attrs=attrs)
        if el and el.get("content"):
            dt = dateparser.parse(el["content"], settings={"RETURN_AS_TIMEZONE_AWARE": False})
            if dt:
                return pd.to_datetime(dt)
    t = soup.find("time")
    if t and (t.get("datetime") or t.text):
        candidate = t.get("datetime") or t.text
        dt = dateparser.parse(candidate, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        if dt:
            return pd.to_datetime(dt)
    return None

def extract_readable_text(html: str) -> str:
    # Best-effort main-text extraction
    if trafilatura is not None:
        try:
            txt = trafilatura.extract(html, include_comments=False)
            if txt and len(txt.split()) > 40:
                return txt
        except Exception:
            pass
    if Document is not None:
        try:
            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), "lxml")
            txt = soup.get_text(" ")
            if txt and len(txt.split()) > 40:
                return txt
        except Exception:
            pass
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "noscript", "header", "footer", "form", "nav"]):
        bad.decompose()
    return soup.get_text(" ")

def normalize_to_weeks(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if u.startswith("week") or u.startswith("wk"):
        return float(val)
    return float(val) * MONTH_TO_WEEKS

# ------------------------- LLM Extraction -------------------------
def get_openai_client():
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not found. Install with `pip install openai` (version >= 1.0)."
        )
    load_dotenv()  # loads .env if present (expects OPENAI_API_KEY)
    return OpenAI()

EXTRACTION_SYSTEM = (
    "You are an information extraction assistant for construction supply chain research. "
    "Extract only facts present in the text."
)

# IMPORTANT: JSON braces are doubled to avoid .format() KeyError
EXTRACTION_USER_TMPL = (
    "You will be given an article text about electrical switchgear and construction.\n"
    "Goal: extract every explicit LEAD TIME mention for switchgear procurement or delivery.\n\n"
    "Return JSON with the shape: {{\"items\": [ ... ]}}\n\n"
    "Each item must have keys:\n"
    "  raw_quote, lead_time_min, lead_time_max, unit, as_weeks_min, as_weeks_max, year, confidence\n"
    "- If a single value is given (e.g., '26 weeks'), set min==max.\n"
    "- Units may be 'weeks' or 'months'. Convert months to weeks using 4.33 and round to 2 decimals.\n"
    "- Determine 'year' using (1) provided article_date, (2) explicit year near the quote, (3) context.\n"
    "- Focus on electrical switchgear in construction. Ignore unrelated lead times.\n\n"
    "Context: article_date={article_date}; url={url}\n\n"
    "Text:\n{text}\n"
)

def _parse_llm_json(text: str) -> List[dict]:
    """Be forgiving about the returned JSON: accept list or dict under common keys."""
    def _coerce_to_items(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "items" in obj and isinstance(obj["items"], list):
                return obj["items"]
            for k in ["extractions", "results", "mentions", "lead_times", "data"]:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
            lists = [v for v in obj.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]
        return []

    try:
        j = json.loads(text)
        return _coerce_to_items(j)
    except Exception:
        pass

    m = re.search(r"\[\s*{.*}\s*\]", text, flags=re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return arr if isinstance(arr, list) else []
        except Exception:
            pass

    m = re.search(r"{.*}", text, flags=re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            return _coerce_to_items(j)
        except Exception:
            pass

    return []

def llm_extract(client, model: str, text: str, url: str, article_date: Optional[pd.Timestamp],
                save_path: Optional[str] = None) -> List[dict]:
    max_chars = 12000
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]
    all_items: List[dict] = []

    for idx, chunk in enumerate(chunks, 1):
        safe_chunk = chunk.replace("{", "{{").replace("}", "}}")  # guard .format on text
        user_msg = EXTRACTION_USER_TMPL.format(
            article_date=(article_date.isoformat() if article_date is not None else None),
            url=url,
            text=safe_chunk,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = resp.choices[0].message.content or ""
            parsed = _parse_llm_json(content)
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                base = re.sub(r"[^a-zA-Z0-9]+", "_", urlparse(url).netloc)[:40]
                with open(os.path.join(save_path, f"{base}_chunk{idx}.json"), "w", encoding="utf-8") as f:
                    f.write(content)
            if parsed:
                all_items.extend(parsed)

        except Exception as e:
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                base = re.sub(r"[^a-zA-Z0-9]+", "_", urlparse(url).netloc)[:40]
                with open(os.path.join(save_path, f"{base}_chunk{idx}_ERROR.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))
            continue

    return all_items

# ------------------------- Regex Fallback -------------------------
def _get_context(text: str, start: int, end: int, window: int = 120) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return re.sub(r"\s+", " ", snippet).strip()

def _infer_year(article_date: Optional[pd.Timestamp], context: str) -> Optional[int]:
    if article_date is not None and article_date.year in TARGET_YEARS:
        return int(article_date.year)
    m = YEAR_NEARBY.search(context)
    if m:
        y = int(m.group(1))
        return y if y in TARGET_YEARS else None
    return None

def regex_fallback(text: str, url: str, article_date: Optional[pd.Timestamp]) -> List[dict]:
    items: List[dict] = []
    for pat in [LT_PAT_1, LT_PAT_2, LT_PAT_3, LT_PAT_4]:
        for m in pat.finditer(text):
            gd = m.groupdict()
            unit = (gd.get("unit") or "").lower()
            v1 = gd.get("val1")
            v2 = gd.get("val2") or v1
            try:
                v1f = float(v1) if v1 else None
                v2f = float(v2) if v2 else None
            except Exception:
                v1f = v2f = None
            if v1f is None:
                continue
            wmin = normalize_to_weeks(v1f, unit)
            wmax = normalize_to_weeks(v2f, unit) if v2f is not None else wmin
            ctx = _get_context(text, *m.span())
            year = _infer_year(article_date, ctx)
            items.append({
                "raw_quote": m.group(0),
                "lead_time_min": v1f,
                "lead_time_max": v2f,
                "unit": unit,
                "as_weeks_min": wmin,
                "as_weeks_max": wmax,
                "year": year,
                "confidence": 0.5,
            })
    return items

# ------------------------- Checkpointing + Logging -------------------------
def ckpt_paths(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    return {
        "jsonl": os.path.join(base_dir, "extractions.jsonl"),
        "processed": os.path.join(base_dir, "processed_urls.txt"),
        "status": os.path.join(base_dir, "status.json"),
        "live_extractions": os.path.join(base_dir, "extractions_live.csv"),
        "live_status": os.path.join(base_dir, "status_live.csv"),
        "log": os.path.join(base_dir, "status.log"),
    }

def log_line(paths: dict, msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(paths["log"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_processed(processed_path: str) -> set:
    if not os.path.exists(processed_path):
        return set()
    with open(processed_path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

def append_processed(processed_path: str, url: str):
    with open(processed_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")

def append_jsonl(jsonl_path: str, records: List[dict]):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_all_records(jsonl_path: str) -> pd.DataFrame:
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

def write_live_files(df: pd.DataFrame, status: dict, live_extractions: str, live_status: str, status_json: str):
    # Always write CSV headers so you can open midway
    if df.empty:
        df = pd.DataFrame(columns=[
            "url", "article_date", "raw_quote", "unit",
            "lead_time_min", "lead_time_max", "weeks_min", "weeks_max",
            "year", "confidence", "origin"
        ])

    def _safe_to_csv(frame: pd.DataFrame, path: str, attempts: int = 5, sleep_s: float = 0.5):
        tmp = f"{path}.tmp"
        for _ in range(attempts):
            try:
                frame.to_csv(tmp, index=False, encoding="utf-8")
                os.replace(tmp, path)  # atomic replace
                return
            except PermissionError:
                time.sleep(sleep_s)
            except Exception:
                # best-effort fallback; don't crash the run
                return
        # clean up tmp if still present
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    _safe_to_csv(df, live_extractions)
    _safe_to_csv(pd.DataFrame([status]), live_status)

    try:
        with open(status_json, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except PermissionError:
        # ignore lock on JSON too
        pass
    except Exception:
        pass

# ------------------------- Helpers for year mapping -------------------------
_URL_YEAR_RE = re.compile(r"/(20\d{2})(?:/|[-_])")

def _year_from_url(url: str) -> Optional[int]:
    """Pick year from URL path like /2023/... if within TARGET_YEARS."""
    m = _URL_YEAR_RE.search(url)
    if not m:
        return None
    y = int(m.group(1))
    return y if y in TARGET_YEARS else None

def _mode_year_targeted(series: pd.Series) -> Optional[int]:
    """Most frequent year among TARGET_YEARS; tie-break by most recent."""
    s = series.dropna().astype(int)
    s = s[s.isin(TARGET_YEARS)]
    if s.empty:
        return None
    vc = s.value_counts()
    maxc = vc.max()
    candidates = sorted(vc[vc == maxc].index, reverse=True)  # prefer newer
    return int(candidates[0])

# ------------------------- Range selection helpers (fixed) -------------------------
def _pick_min_range(df_mentions: pd.DataFrame) -> Tuple[Optional[float], Optional[float], int]:
    """Pick the mention with the smallest w_low. Return (low, high) and unique URL count at that exact pair."""
    if df_mentions.empty or "w_low" not in df_mentions or "w_high" not in df_mentions:
        return None, None, 0
    i = df_mentions["w_low"].idxmin()
    low = float(df_mentions.at[i, "w_low"])
    high = float(df_mentions.at[i, "w_high"])
    tol = 1e-6
    pair_matches = df_mentions[np.isclose(df_mentions["w_low"], low, atol=tol) &
                               np.isclose(df_mentions["w_high"], high, atol=tol)]
    urls_at_pair = pair_matches["url"].nunique()
    return low, high, int(urls_at_pair)

def _pick_max_range(df_mentions: pd.DataFrame) -> Tuple[Optional[float], Optional[float], int]:
    """Pick the mention with the largest w_high. Return (low, high) and unique URL count at that exact pair."""
    if df_mentions.empty or "w_low" not in df_mentions or "w_high" not in df_mentions:
        return None, None, 0
    i = df_mentions["w_high"].idxmax()
    low = float(df_mentions.at[i, "w_low"])
    high = float(df_mentions.at[i, "w_high"])
    tol = 1e-6
    pair_matches = df_mentions[np.isclose(df_mentions["w_low"], low, atol=tol) &
                               np.isclose(df_mentions["w_high"], high, atol=tol)]
    urls_at_pair = pair_matches["url"].nunique()
    return low, high, int(urls_at_pair)

# ------------------------- Output writer -------------------------
def write_final_excel(paths: dict, urls: List[str], out_path: str):
    """Build the single required table (min/avg/max) for 2021–2024 and write it to Excel."""
    df_all = load_all_records(paths["jsonl"])

    # Base frame of all URLs
    per_url = pd.DataFrame({"url": urls})

    # Collect article_date and mention-level years (if we have any records)
    if not df_all.empty:
        df_all["article_date"] = pd.to_datetime(df_all["article_date"], errors="coerce")
        df_all["w_low"] = df_all[["weeks_min", "weeks_max"]].min(axis=1, skipna=True)
        df_all["w_high"] = df_all[["weeks_min", "weeks_max"]].max(axis=1, skipna=True)

        # First non-null article_date per URL
        agg_dates = (
            df_all.groupby("url", as_index=False)
            .agg(article_date=("article_date", lambda s: next((v for v in s if pd.notna(v)), pd.NaT)))
        )
        per_url = per_url.merge(agg_dates, on="url", how="left")

        # Mode of mention-year per URL (as a backstop)
        mode_yr = (
            df_all.groupby("url", as_index=False)
            .agg(mention_year=("year", _mode_year_targeted))
        )
        per_url = per_url.merge(mode_yr, on="url", how="left")
    else:
        per_url["article_date"] = pd.NaT
        per_url["mention_year"] = np.nan

    # Year from URL path
    per_url["url_year"] = per_url["url"].map(_year_from_url)

    # Derive article-year from metadata if within target
    per_url["article_year"] = pd.to_datetime(per_url["article_date"], errors="coerce").dt.year
    per_url.loc[~per_url["article_year"].isin(TARGET_YEARS), "article_year"] = np.nan

    # Final mapped published_year with strict fallback order
    per_url["published_year"] = per_url["article_year"].combine_first(per_url["url_year"]).combine_first(per_url["mention_year"])

    YEARS = TARGET_YEARS

    # Count of URLs per mapped year
    url_counts = (
        per_url[per_url["published_year"].isin(YEARS)]
        .groupby("published_year", as_index=False)
        .agg(urls_count=("url", "nunique"))
        .rename(columns={"published_year": "year"})
        .set_index("year")
        .reindex(YEARS, fill_value=0)
        .reset_index()
    )

    # Lead-time mentions aligned to mapped year
    if not df_all.empty:
        df_all_yr = df_all.merge(
            per_url[["url", "published_year"]],
            on="url",
            how="left"
        )
        df_all_yr = df_all_yr[df_all_yr["published_year"].isin(YEARS)].copy()
    else:
        df_all_yr = pd.DataFrame(columns=["url", "published_year", "w_low", "w_high"])

    # Stats per year
    rows = []
    for y in YEARS:
        urls_total = int(url_counts.loc[url_counts["year"] == y, "urls_count"].iloc[0]) if not url_counts.empty else 0

        mentions_y = df_all_yr[df_all_yr["published_year"] == y][["url", "w_low", "w_high"]].dropna()
        urls_with_mentions = mentions_y["url"].nunique() if not mentions_y.empty else 0

        # Minimum range: pick the mention with smallest w_low; pair with its w_high
        min_low, min_high, urls_at_min = _pick_min_range(mentions_y) if not mentions_y.empty else (None, None, 0)

        # Maximum range: pick the mention with largest w_high; pair with its w_low
        max_low, max_high, urls_at_max = _pick_max_range(mentions_y) if not mentions_y.empty else (None, None, 0)

        # Average range: compute per-URL averages first to avoid overweighting verbose pages
        if not mentions_y.empty:
            per_url_avg = (
                mentions_y.groupby("url", as_index=False)
                .agg(avg_low=("w_low", "mean"), avg_high=("w_high", "mean"))
            )
            avg_low = float(per_url_avg["avg_low"].mean())
            avg_high = float(per_url_avg["avg_high"].mean())
            urls_used_for_avg = int(per_url_avg["url"].nunique())
        else:
            avg_low = None
            avg_high = None
            urls_used_for_avg = 0

        def _r(v):
            return None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else round(float(v), 1)

        rows.append({
            "year": y,
            "urls_count": int(urls_total),
            "urls_with_mentions": int(urls_with_mentions),
            "min_low_weeks": _r(min_low),
            "min_high_weeks": _r(min_high),
            "urls_at_min": int(urls_at_min),
            "avg_low_weeks": _r(avg_low),
            "avg_high_weeks": _r(avg_high),
            "urls_used_for_avg": int(urls_used_for_avg),
            "max_low_weeks": _r(max_low),
            "max_high_weeks": _r(max_high),
            "urls_at_max": int(urls_at_max),
        })

    table = pd.DataFrame(rows)

    # Write a single sheet with just this table
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        table.to_excel(xw, index=False, sheet_name="lead_time_ranges_by_year")

    # Live CSVs status
    try:
        done_count = len(load_processed(paths["processed"]))
    except Exception:
        done_count = 0
    status = {"processed": done_count}
    write_live_files(df_all, status, paths["live_extractions"], paths["live_status"], paths["status"])

    # Console note about unmapped URLs so you know what remains unspecified
    unmapped = int((~per_url["published_year"].isin(YEARS)).sum())
    if unmapped:
        log_line(paths, f"Note: {unmapped} URLs could not be confidently mapped to 2021–2024.")

# ------------------------- Main Pipeline -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="switchgear_summaries.csv",
                    help="CSV file containing URLs (default: switchgear_summaries.csv)")
    ap.add_argument("--out", default="switchgear_lead_time_ranges.xlsx",
                    help="Final Excel file path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--delay", type=float, default=1.0,
                    help="Seconds to sleep between URLs")
    ap.add_argument("--verbose", action="store_true",
                    help="Print step-by-step logs for each URL")
    ap.add_argument("--save-llm-dir", default=None,
                    help="Directory to save raw LLM JSON outputs")
    ap.add_argument("--checkpoint-dir", default="checkpoint_switchgear",
                    help="Directory for resumable checkpoints and live CSVs")
    ap.add_argument("--resume", action="store_true",
                    help="Force resume mode (default behavior)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Disable resume; process all URLs from scratch")
    ap.add_argument("--offline", action="store_true",
                    help="Do not call OpenAI; use regex fallback only")
    ap.add_argument("--sync-every", type=int, default=0,
                    help="Write Excel every N processed URLs (0 = only at end)")
    args = ap.parse_args()

    paths = ckpt_paths(args.checkpoint_dir)
    out_abs = os.path.abspath(args.out)

    # Prepare OpenAI only if online mode
    client = None
    if not args.offline:
        client = get_openai_client()

    urls = read_urls_csv(args.input)

    # Resume logic: default True unless --no-resume was passed
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = True  # default

    processed = load_processed(paths["processed"]) if resume else set()
    total = len(urls)

    # Start banner
    log_line(paths, f"Run started | urls={total} | resume={resume} | processed_already={len(processed)} | out={out_abs} | offline={bool(args.offline)}")

    start_time = time.time()

    def _status_dict(done: int) -> dict:
        elapsed = time.time() - start_time
        return {
            "processed": done,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "remaining_est_sec": round(elapsed / max(done, 1) * (total - done), 2) if done else None,
            "offline_mode": bool(args.offline),
            "resume_mode": bool(resume),
        }

    write_live_files(pd.DataFrame(), _status_dict(0), paths["live_extractions"], paths["live_status"], paths["status"])

    # Signal handling to save on interrupt
    def _on_term(signame):
        log_line(paths, f"Signal {signame} received; writing final Excel...")
        try:
            write_final_excel(paths, urls, out_abs)
        finally:
            log_line(paths, f"Final Excel written -> {out_abs}. Exiting.")
            sys.exit(130)

    def _sig_handler(sig, frame):
        name = {getattr(signal, "SIGINT", None): "SIGINT",
                getattr(signal, "SIGTERM", None): "SIGTERM",
                getattr(signal, "SIGBREAK", None): "SIGBREAK"}.get(sig, "SIGNAL")
        _on_term(name)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass
    for maybe in ("SIGTERM", "SIGBREAK"):
        s = getattr(signal, maybe, None)
        if s is not None:
            try:
                signal.signal(s, _sig_handler)
            except Exception:
                pass

    atexit.register(lambda: write_final_excel(paths, urls, out_abs))

    processed_count = 0 if not resume else len(processed)

    try:
        with tqdm(total=total, unit="url", desc="Processing URLs", initial=processed_count) as pbar:
            for idx, url in enumerate(urls, 1):
                if resume and (url in processed):
                    pbar.update(1)
                    continue

                domain = urlparse(url).netloc or "unknown"
                pbar.set_postfix({"host": domain[:30]})

                if args.verbose:
                    pbar.write(f"[{idx}/{total}] Fetching: {url}")

                html = fetch_html(url)
                if not html:
                    if args.verbose:
                        pbar.write(f"[{idx}/{total}] Fetch failed for {domain}")
                    append_processed(paths["processed"], url)
                    pbar.update(1)
                    time.sleep(args.delay)
                    df_live = load_all_records(paths["jsonl"])
                    write_live_files(df_live, _status_dict(pbar.n), paths["live_extractions"], paths["live_status"], paths["status"])
                    if args.sync_every and (pbar.n % args.sync_every == 0):
                        write_final_excel(paths, urls, out_abs)
                    continue

                article_date = extract_meta_date(html)
                if args.verbose:
                    pbar.write(f"[{idx}/{total}] Extracting main text (date={article_date})")

                text = extract_readable_text(html)
                if not text or len(text.split()) < 40:
                    if args.verbose:
                        pbar.write(f"[{idx}/{total}] Not enough text for {domain}")
                    append_processed(paths["processed"], url)
                    pbar.update(1)
                    time.sleep(args.delay)
                    df_live = load_all_records(paths["jsonl"])
                    write_live_files(df_live, _status_dict(pbar.n), paths["live_extractions"], paths["live_status"], paths["status"])
                    if args.sync_every and (pbar.n % args.sync_every == 0):
                        write_final_excel(paths, urls, out_abs)
                    continue

                if args.verbose:
                    mode = "REGEX-only (offline)" if args.offline else "LLM + regex fallback"
                    pbar.write(f"[{idx}/{total}] Analyzing ({mode}) for {domain}")

                records = []
                used_llm = False
                extractions: List[dict] = []

                # Online (LLM) first unless offline
                if not args.offline and client is not None:
                    llm_items = llm_extract(client, args.model, text, url, article_date, save_path=args.save_llm_dir)
                    if llm_items:
                        extractions = llm_items
                        used_llm = True

                # If nothing from LLM or offline, run regex
                if not extractions:
                    extractions = regex_fallback(text, url, article_date)
                    used_llm = False

                if args.verbose:
                    pbar.write(f"[{idx}/{total}] {domain} -> {len(extractions)} lead-time mentions")

                for it in extractions:
                    unit = (it.get("unit") or "").lower()
                    lt_min = it.get("lead_time_min")
                    lt_max = it.get("lead_time_max") if it.get("lead_time_max") is not None else lt_min
                    as_weeks_min = it.get("as_weeks_min")
                    as_weeks_max = it.get("as_weeks_max")

                    def _conv(v):
                        if v is None:
                            return None
                        try:
                            v = float(v)
                        except Exception:
                            return None
                        if unit.startswith("week"):
                            return float(v)
                        return float(v) * MONTH_TO_WEEKS

                    if as_weeks_min is None:
                        as_weeks_min = _conv(lt_min)
                    if as_weeks_max is None:
                        as_weeks_max = _conv(lt_max)

                    year = it.get("year")
                    try:
                        year = int(year) if year is not None else None
                    except Exception:
                        year = None

                    records.append({
                        "url": url,
                        "article_date": article_date.isoformat() if article_date is not None else None,
                        "raw_quote": it.get("raw_quote"),
                        "unit": unit,
                        "lead_time_min": lt_min,
                        "lead_time_max": lt_max,
                        "weeks_min": as_weeks_min,
                        "weeks_max": as_weeks_max,
                        "year": year,
                        "confidence": it.get("confidence"),
                        "origin": ("llm" if used_llm else "regex"),
                    })

                if records:
                    append_jsonl(paths["jsonl"], records)

                append_processed(paths["processed"], url)

                df_live = load_all_records(paths["jsonl"])
                write_live_files(df_live, _status_dict(pbar.n + 1), paths["live_extractions"], paths["live_status"], paths["status"])

                if args.sync_every and ((pbar.n + 1) % args.sync_every == 0):
                    write_final_excel(paths, urls, out_abs)

                pbar.update(1)
                time.sleep(args.delay)

    except KeyboardInterrupt:
        log_line(paths, "KeyboardInterrupt detected; writing final Excel from checkpoints...")

    finally:
        write_final_excel(paths, urls, out_abs)

        try:
            df_table = pd.read_excel(out_abs, sheet_name="lead_time_ranges_by_year")
        except Exception:
            df_table = pd.DataFrame()
        log_line(paths, f"Saved final Excel: {out_abs}")
        if not df_table.empty:
            try:
                log_line(paths, "lead_time_ranges_by_year (head):")
                print(df_table.head(10).to_string(index=False), flush=True)
            except Exception:
                pass
        log_line(paths, "Run complete.")

if __name__ == "__main__":
    main()
