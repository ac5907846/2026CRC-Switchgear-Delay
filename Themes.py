#!/usr/bin/env python3
"""
Step 4 - Per-row Theming for 10-15 Buckets (gpt-4o-mini)

- One theme per ROW per COLUMN (so totals = number of rows, e.g., 414)
- Consolidates to ~10-15 themes by summarizing and merging near-duplicates
- Falls back to a minimal "Other (Long Tail)" only if needed and capped by percent
- Caches row->theme assignments so you can restart safely
- Uses OpenAI gpt-4o-mini with JSON Schema output
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from slugify import slugify
from difflib import SequenceMatcher

# -------------------------- Helpers --------------------------

MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(s: Any) -> str:
    s = ("" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)).strip()
    s = s.strip('"\'[](){}')
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip(" .,")


def ensure_csv_path(csv_arg: str) -> Path:
    p = Path(csv_arg)
    if p.exists():
        return p
    p2 = Path(str(p) + ".csv")
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {csv_arg} or {csv_arg}.csv")


def file_md5(path: Path, chunk_size: int = 1_000_000) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def make_dataset_id(csv_path: Path) -> str:
    md5 = file_md5(csv_path)
    size = csv_path.stat().st_size
    return f"md5{md5[:12]}_sz{size}"


def auto_detect_columns(
    df: pd.DataFrame, explicit: Optional[List[str]], exclude: Optional[List[str]]
) -> List[str]:
    def resolve(names: List[str]) -> List[str]:
        out: List[str] = []
        for want in names:
            want_l = want.strip().lower()
            hit = None
            for c in df.columns:
                c_l = c.strip().lower()
                if c_l == want_l or want_l in c_l:
                    hit = c
                    break
            if hit and hit not in out:
                out.append(hit)
        return out

    cols = (
        resolve(explicit)
        if explicit
        else [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    )

    if exclude:
        ex = [e.strip().lower() for e in exclude]
        cols = [c for c in cols if all(e not in c.strip().lower() for e in ex)]
    return cols


@dataclass
class CacheKey:
    dataset_id: str
    column: str
    params_key: str

    def stem(self) -> str:
        return f"{self.dataset_id}__{slugify(self.column)}__{self.params_key}"


# -------- Best-fix caching: hashed filenames + legacy compatibility --------

def _hash_stem(stem: str) -> str:
    return hashlib.md5(stem.encode("utf-8")).hexdigest()[:16]

def _cache_files(cache_dir: Path, ck: CacheKey) -> Tuple[Path, Path]:
    short = _hash_stem(ck.stem())
    rowmap = cache_dir / f"{short}__rowmap.jsonl"
    themes = cache_dir / f"{short}__themes.json"
    return rowmap, themes

def _legacy_cache_files(cache_dir: Path, ck: CacheKey) -> Tuple[Path, Path]:
    return (
        cache_dir / f"{ck.stem()}__rowmap.jsonl",
        cache_dir / f"{ck.stem()}__themes.json",
    )

# -------------------------- OpenAI schema + prompts --------------------------

ASSIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "theme": {"type": "string"}},
                "required": ["id", "theme"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assignments", "themes"],
    "additionalProperties": False,
}

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                "required": ["from", "to"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["canonical_themes", "mapping"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You assign exactly one theme to each row of text from a CSV column.\n"
    "Rules:\n"
    "1) Return valid JSON for the provided schema.\n"
    "2) Assign exactly one theme to each item.\n"
    "3) Use short, neutral, Title Case names (<= 5 words).\n"
    "4) Prefer reusing any existing theme names provided.\n"
    "5) Keep the number of distinct themes near the requested target; avoid overly specific names.\n"
    "6) If the column looks like lead time/duration, normalize to concise buckets (e.g., 0-4 wks, 5-8 wks, 9-12 wks, 13-16 wks, >16 wks).\n"
)

CONSOLIDATE_PROMPT = (
    "You are consolidating a list of theme names and their frequencies into a small set of canonical, "
    "distinct, Title Case themes suitable for a summary table.\n"
    "Requirements:\n"
    "1) Produce AT MOST {max_themes} canonical themes, NO 'Other'.\n"
    "2) Map EVERY original theme name to exactly one canonical theme via 'mapping'.\n"
    "3) Prefer broader, neutral categories and merge near-duplicates/specific subtypes accordingly.\n"
    "4) Keep names short (<= 5 words) and consistent across the set.\n"
    "5) If lead-time buckets are present, keep them normalized (0-4 wks, 5-8 wks, 9-12 wks, 13-16 wks, >16 wks).\n"
    "6) Return valid JSON for the provided schema."
)


def get_client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file next to this script.")
    return OpenAI()

# -------------------------- ROBUST JSON HANDLER (revised) --------------------------

def _responses_create_json(client: OpenAI, model: str, system: str, user_payload: dict, schema: dict) -> dict:
    def _safe_load(s: str) -> dict:
        if not s or not isinstance(s, str):
            return {}
        s = s.strip()
        # Strip simple code fences if present
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
        # Try direct JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try largest {...} slice
        try:
            m = re.search(r"\{.*\}", s, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return {}

    try:
        # Preferred: Responses API with structured outputs
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "ToolSchema", "schema": schema, "strict": True}},
            temperature=0,
        )
        # Try output_text first
        text = getattr(resp, "output_text", None)
        if not text:
            # Fall back to streaming-style pieces if present
            chunks = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", ""))
            text = "".join(chunks) if chunks else ""
        return _safe_load(text)
    except Exception:
        # Fallback: old Chat Completions path, also fully guarded
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system + "\nReturn ONLY valid JSON conforming to the described fields."},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0,
            )
            text = (completion.choices[0].message.content or "")
            return _safe_load(text)
        except Exception:
            # Absolute last resort: return empty so callers can fall back deterministically
            return {}

# -------------------------- Coercion helpers --------------------------

def gpt_assign_batch(
    client: OpenAI,
    model: str,
    column_name: str,
    items: List[Tuple[int, str]],
    existing_theme_names: List[str],
    target_themes: int,
) -> dict:
    payload = {
        "column": column_name,
        "target_themes": target_themes,
        "existing_theme_names": existing_theme_names,
        "items": [{"id": i, "text": t} for i, t in items],
        "instructions": f"Assign one theme to each item; keep total themes close to {target_themes}.",
    }
    try:
        return _responses_create_json(client, model, SYSTEM_PROMPT, payload, ASSIGN_SCHEMA)
    except TypeError:
        # Extra fallback (older SDKs)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nReturn ONLY valid JSON per the schema."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
        )
        text = completion.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {"assignments": [], "themes": []}


def _coerce_mapping_pairs(mapping_list: Any) -> List[Tuple[str, str]]:
    """
    Accepts:
      - [{'from': 'old', 'to': 'new'}, ...]
      - [['old','new'], ...] or tuples
      - ['old -> new', 'A=>B', 'X:Y', '{"from":"A","to":"B"}', ...]
    Returns list of (from, to) pairs as strings.
    """
    pairs: List[Tuple[str, str]] = []
    if not isinstance(mapping_list, list):
        return pairs
    for m in mapping_list:
        if isinstance(m, dict):
            frm = m.get("from") or m.get("source") or m.get("orig") or m.get("name")
            to = m.get("to") or m.get("target") or m.get("canonical") or m.get("map_to")
            if frm and to:
                pairs.append((str(frm), str(to)))
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            pairs.append((str(m[0]), str(m[1])))
        elif isinstance(m, str):
            s = m.strip()
            # Try JSON object inside a string
            try:
                o = json.loads(s)
                if isinstance(o, dict):
                    frm = o.get("from") or o.get("source") or o.get("orig") or o.get("name")
                    to = o.get("to") or o.get("target") or o.get("canonical") or o.get("map_to")
                    if frm and to:
                        pairs.append((str(frm), str(to)))
                        continue
            except Exception:
                pass
            # Try common separators
            for sep in ["->", "=>", ":"]:
                if sep in s:
                    a, b = [normalize_text(p) for p in s.split(sep, 1)]
                    if a and b:
                        pairs.append((a, b))
                    break
    return pairs


def _coerce_canonical_names(canonical_list: Any) -> List[str]:
    """
    Accepts:
      - [{'name':'X', 'description':'...'}, ...]
      - ['X','Y', ...]
      - [['X', ...], ...] (take first element)
    Returns list of canonical names (strings).
    """
    out: List[str] = []
    if not isinstance(canonical_list, list):
        return out
    for c in canonical_list:
        nm = ""
        if isinstance(c, dict):
            nm = normalize_text(c.get("name", ""))
        elif isinstance(c, (list, tuple)) and len(c) >= 1:
            nm = normalize_text(c[0])
        else:
            nm = normalize_text(c)
        if nm:
            out.append(nm)
    return [n for n in out if n]


# -------------------------- CONSOLIDATION (revised) --------------------------

def consolidate_theme_names(
    client: OpenAI,
    model: str,
    counts: Counter,
    max_themes: int,
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Returns a robust mapping from original theme -> canonical theme and a
    list of canonical theme dicts. Never raises; falls back deterministically.
    """
    if not counts:
        return {}, []

    payload = {
        "max_themes": max_themes,
        "themes": [{"name": n, "freq": int(f)} for n, f in sorted(counts.items(), key=lambda x: (-x[1], x[0]))],
    }

    data = _responses_create_json(
        client,
        model,
        CONSOLIDATE_PROMPT.format(max_themes=max_themes),
        payload,
        CONSOLIDATE_SCHEMA,
    ) or {}

    # Defensive coercion
    mapping_pairs = _coerce_mapping_pairs(data.get("mapping", []))
    canonical_names = _coerce_canonical_names(data.get("canonical_themes", []))

    # If model didn't supply canonical names, keep the top max_themes originals
    if not canonical_names:
        canonical_names = [n for n, _ in counts.most_common(max_themes)]

    # If still empty (extreme case), synthesize from counts keys
    if not canonical_names:
        canonical_names = list(counts.keys())[:max_themes]

    # Build initial mapping from pairs; ignore bad items
    name_map: Dict[str, str] = {}
    for frm, to in mapping_pairs:
        frm_n = normalize_text(frm)
        to_n = normalize_text(to)
        if frm_n and to_n:
            name_map[frm_n] = to_n

    # Ensure coverage for every original theme name
    for n in counts.keys():
        if n not in name_map:
            # Closest canonical by similarity, else first canonical
            best = None
            best_r = -1.0
            for cn in canonical_names:
                r = SequenceMatcher(None, n.lower(), cn.lower()).ratio()
                if r > best_r:
                    best_r = r
                    best = cn
            name_map[n] = best or canonical_names[0]

    # Build canonical defs with optional descriptions if present
    by_name: Dict[str, Dict[str, str]] = {}
    if isinstance(data.get("canonical_themes"), list):
        for t in data["canonical_themes"]:
            if isinstance(t, dict):
                nm = normalize_text(t.get("name", ""))
                if nm and nm not in by_name:
                    by_name[nm] = {"name": nm, "description": normalize_text(t.get("description", ""))}
    for nm in canonical_names:
        by_name.setdefault(nm, {"name": nm, "description": ""})

    return name_map, list(by_name.values())


# -------------------------- Caching (row -> theme) --------------------------

def cache_read(cache_dir: Path, ck: CacheKey) -> Tuple[Dict[int, str], List[Dict[str, str]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_map_file, theme_defs_file = _cache_files(cache_dir, ck)
    legacy_rowmap, legacy_themes = _legacy_cache_files(cache_dir, ck)

    row_to_theme: Dict[int, str] = {}
    theme_defs: List[Dict[str, str]] = []

    row_src = row_map_file if row_map_file.exists() else legacy_rowmap
    th_src = theme_defs_file if theme_defs_file.exists() else legacy_themes

    if row_src.exists():
        with open(row_src, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "id" in rec and "theme" in rec:
                        row_to_theme[int(rec["id"])] = rec["theme"]
                except Exception:
                    continue
        if row_src is legacy_rowmap and not row_map_file.exists():
            row_map_file.parent.mkdir(parents=True, exist_ok=True)
            with open(row_map_file, "w", encoding="utf-8") as out:
                for rid, th in sorted(row_to_theme.items()):
                    out.write(json.dumps({"id": int(rid), "theme": th}, ensure_ascii=False) + "\n")

    if th_src.exists():
        try:
            theme_defs = json.loads(th_src.read_text(encoding="utf-8"))
            if th_src is legacy_themes and not theme_defs_file.exists():
                theme_defs_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = theme_defs_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(theme_defs, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(theme_defs_file)
        except Exception:
            theme_defs = []

    return row_to_theme, theme_defs


def cache_append_rowmap(cache_dir: Path, ck: CacheKey, batch_map: Dict[int, str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_map_file, _ = _cache_files(cache_dir, ck)
    row_map_file.parent.mkdir(parents=True, exist_ok=True)
    with open(row_map_file, "a", encoding="utf-8") as out:
        for rid, th in batch_map.items():
            out.write(json.dumps({"id": int(rid), "theme": th}, ensure_ascii=False) + "\n")


def cache_write_themes(cache_dir: Path, ck: CacheKey, theme_defs: List[Dict[str, str]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, theme_defs_file = _cache_files(cache_dir, ck)
    theme_defs_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = theme_defs_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(theme_defs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(theme_defs_file)


# -------------------------- Data collection (one item per row) --------------------------

def collect_rows_by_column(
    df: pd.DataFrame, columns: List[str], total_websites: int, blank_label: str
) -> Dict[str, List[str]]:
    per_col: Dict[str, List[str]] = {c: [] for c in columns}
    total_rows = len(df)
    print(f"Loaded {total_rows} rows; expecting {total_websites} websites.")
    print("Scanning rows - one item per row")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for c in columns:
            val = normalize_text(row[c])
            per_col[c].append(val if val else blank_label)
        print(f"{i} out of {total_websites} complete", end="\r")
    print(f"{min(total_rows, total_websites)} out of {total_websites} complete")
    return per_col


# -------------------------- Per-column processing --------------------------

def process_column(
    client: OpenAI,
    model: str,
    texts: List[str],
    col: str,
    cache_dir: Path,
    dataset_id: str,
    force: bool,
    batch_size: int,
    target_themes: int,
    min_themes: int,
    max_themes: int,
    params_key: str,
    max_other_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      themes_table: Theme | Frequency | Percent
      raw_table: Row | Theme
    """
    ck = CacheKey(dataset_id=dataset_id, column=col, params_key=params_key)
    row_to_theme, theme_defs = cache_read(cache_dir, ck)
    existing_names = [t.get("name", "") for t in theme_defs if t.get("name")]

    n = len(texts)

    if force:
        row_to_theme = {}
        theme_defs = []
        existing_names = []
        for f in (*_cache_files(cache_dir, ck), *_legacy_cache_files(cache_dir, ck)):
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    todo_ids = [i for i in range(n) if i not in row_to_theme]
    total_batches = (len(todo_ids) + batch_size - 1) // batch_size if batch_size > 0 else 0

    b = 0
    i = 0
    while i < len(todo_ids):
        batch_ids = todo_ids[i : i + batch_size]
        i += batch_size
        b += 1
        batch_items = [(rid, texts[rid]) for rid in batch_ids]
        print(f"[{col}] GPT theming - batch {b}/{total_batches} (rows {batch_ids[0]+1}-{batch_ids[-1]+1})")
        data = gpt_assign_batch(client, model, col, batch_items, existing_names, target_themes)

        # Collect assignments
        batch_map: Dict[int, str] = {}
        new_defs: Dict[str, str] = {}
        for a in data.get("assignments", []):
            try:
                rid = int(a.get("id"))
                name = normalize_text(a.get("theme", ""))
                if name:
                    batch_map[rid] = name
            except AttributeError:
                continue

        for t in data.get("themes", []):
            if isinstance(t, dict):
                nm = normalize_text(t.get("name", ""))
                if nm and nm not in new_defs:
                    new_defs[nm] = normalize_text(t.get("description", ""))

        if batch_map:
            cache_append_rowmap(cache_dir, ck, batch_map)
            row_to_theme.update(batch_map)

        # Merge theme defs (keep first description)
        by_name = {t["name"]: t for t in theme_defs if "name" in t}
        for nm, desc in new_defs.items():
            if nm not in by_name:
                by_name[nm] = {"name": nm, "description": desc}
        theme_defs = list(by_name.values())
        existing_names = [t["name"] for t in theme_defs]

        # Fallback: ensure every batch row is assigned
        for rid in batch_ids:
            if rid not in row_to_theme:
                row_to_theme[rid] = texts[rid]
                cache_append_rowmap(cache_dir, ck, {rid: texts[rid]})

        cache_write_themes(cache_dir, ck, theme_defs)

    # Build raw items table
    raw_rows = [{"Row": i + 1, "Theme": row_to_theme.get(i, "Unassigned")} for i in range(n)]
    raw_df = pd.DataFrame(raw_rows)

    # -------- Consolidation phase: collapse near-duplicates to <= max_themes canonical --------
    original_counts = Counter(raw_df["Theme"].tolist())
    name_map, canonical_defs = consolidate_theme_names(client, model, original_counts, max_themes=max_themes)

    # Apply mapping
    raw_df["Theme"] = raw_df["Theme"].map(lambda x: name_map.get(x, x))
    # Recount after consolidation
    counts = Counter(raw_df["Theme"].tolist())

    # Replace theme defs with canonical ones for this column
    theme_defs = canonical_defs or [{"name": t, "description": ""} for t in counts.keys()]
    cache_write_themes(cache_dir, ck, theme_defs)

    # -------- Build final table; only create "Other" if unavoidable and keep it minimal --------
    sorted_pairs = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    total = n

    if len(sorted_pairs) > max_themes:
        # adaptive keep so that 'Other' <= max_other_percent
        keep = max_themes - 1
        while keep < len(sorted_pairs) - 1:
            tail = sorted_pairs[keep:]
            other_total = sum(freq for _, freq in tail)
            if (100.0 * other_total / total) <= max_other_percent:
                break
            keep += 1
        # If keep would consume everything, just keep all and drop Other entirely
        if keep >= len(sorted_pairs):
            final_pairs = sorted_pairs[:max_themes]
        else:
            head = sorted_pairs[:keep]
            tail = sorted_pairs[keep:]
            other_total = sum(freq for _, freq in tail)
            final_pairs = head + [("Other (Long Tail)", other_total)]
    else:
        final_pairs = sorted_pairs

    themes_df = pd.DataFrame(
        [{"Theme": t, "Frequency": f, "Percent": round(100.0 * f / total, 1)} for t, f in final_pairs]
    )

    return themes_df, raw_df


def column_is_lead_time(name: str) -> bool:
    l = name.strip().lower()
    return any(k in l for k in ["lead time", "lead_time", "lead-time", "duration", "delivery"])


def _excel_safe_name(name: str) -> str:
    name = re.sub(r'[][:*?/\\]', "", name)
    return name[:31]


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-row theming with gpt-4o-mini - export frequency tables to Excel. Resumable."
    )
    ap.add_argument("--csv", default="switchgear_summaries", help="Path to CSV (with or without .csv)")
    ap.add_argument("--out", default="theme_frequencies.xlsx", help="Output Excel file path")
    ap.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to process; auto-detect if omitted",
    )
    ap.add_argument(
        "--exclude",
        default="URL,Link",
        help="Comma-separated name fragments to skip",
    )
    ap.add_argument("--cache-dir", default=".gpt_theme_cache", help="Directory for caches (JSONL)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    ap.add_argument("--batch-size", type=int, default=120, help="Rows per GPT request")
    ap.add_argument("--target-themes", type=int, default=12, help="Desired number of themes per column")
    ap.add_argument("--min-themes", type=int, default=10, help="Minimum themes before consolidation")
    ap.add_argument("--max-themes", type=int, default=15, help="Maximum themes; prefer consolidation to this cap")
    ap.add_argument("--max-other-percent", type=float, default=5.0, help="Cap for 'Other' as percent of total")
    ap.add_argument("--total-websites", type=int, default=414, help="Total rows for progress display")
    ap.add_argument("--blank-label", default="Unspecified", help="Label to use for blank cells")
    ap.add_argument("--force", action="store_true", help="Ignore cache for the selected columns")
    args = ap.parse_args()

    csv_path = ensure_csv_path(args.csv)
    df = pd.read_csv(csv_path)

    print("Using OpenAI model:", args.model, "- one theme per row; totals match row count.")

    dataset_id = make_dataset_id(csv_path)
    params_key = hashlib.md5(
        json.dumps(
            {
                "model": args.model,
                "batch_size": args.batch_size,
                "target": args.target_themes,
                "min": args.min_themes,
                "max": args.max_themes,
                "max_other_percent": args.max_other_percent,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    explicit_cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    exclude_cols = [c.strip() for c in args.exclude.split(",")] if args.exclude else None
    columns = auto_detect_columns(df, explicit_cols, exclude_cols)
    if not columns:
        print("No text-like columns detected after filters.")
        sys.exit(1)

    client = get_client()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_col_rows = collect_rows_by_column(
        df, columns, total_websites=args.total_websites, blank_label=args.blank_label
    )

    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=_excel_safe_name("Raw Rows"))

        combined = Counter()

        for col in columns:
            texts = per_col_rows[col]
            print(f"\nTheming column: {col} (rows: {len(texts)})")
            themes_df, raw_df = process_column(
                client,
                args.model,
                texts,
                col,
                cache_dir,
                dataset_id,
                force=args.force,
                batch_size=args.batch_size,
                target_themes=args.target_themes,
                min_themes=args.min_themes,
                max_themes=args.max_themes,
                params_key=params_key,
                max_other_percent=args.max_other_percent,
            )

            safe_col = re.sub(r"[^A-Za-z0-9 _-]", "", col)
            sheet_name = _excel_safe_name("LeadTimeBuckets") if column_is_lead_time(col) else _excel_safe_name(f"Themes - {safe_col}")
            raw_sheet = _excel_safe_name(f"Raw Items - {safe_col}")

            themes_df.to_excel(writer, index=False, sheet_name=sheet_name)
            raw_df.to_excel(writer, index=False, sheet_name=raw_sheet)

            for theme, freq in themes_df[["Theme", "Frequency"]].values.tolist():
                combined[f"{col}: {theme}"] += int(freq)

        comb_df = pd.DataFrame(
            sorted(combined.items(), key=lambda x: (-x[1], x[0])), columns=["Theme", "Frequency"]
        )
        comb_df.to_excel(writer, index=False, sheet_name=_excel_safe_name("Combined Summary"))

        meta = pd.DataFrame(
            [
                {"key": "dataset_id", "value": dataset_id},
                {"key": "csv_path", "value": str(csv_path)},
                {"key": "model", "value": args.model},
                {"key": "batch_size", "value": args.batch_size},
                {"key": "target_themes", "value": args.target_themes},
                {"key": "min_themes", "value": args.min_themes},
                {"key": "max_themes", "value": args.max_themes},
                {"key": "max_other_percent", "value": args.max_other_percent},
                {"key": "columns_processed", "value": ", ".join(columns)},
                {"key": "cache_dir", "value": str(cache_dir)},
                {"key": "params_key", "value": params_key},
                {"key": "total_websites", "value": args.total_websites},
                {"key": "blank_label", "value": args.blank_label},
            ]
        )
        meta.to_excel(writer, index=False, sheet_name=_excel_safe_name("_Meta"))

    print(f"\nWrote {args.out}")
    print("Sheets: Raw Rows, Themes or LeadTimeBuckets per column, Raw Items per column, Combined Summary, _Meta")


if __name__ == "__main__":
    main()
