"""
extract_topics.py
Reads ESGscopus.csv, calls Claude, and writes topic_extraction.csv.
If the run stops for any reason, re-running the script continues where it left off.
"""

import csv
import json
import os
import sys
import time
from typing import Dict, Set

import anthropic  # pip install anthropic==0.24.0

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
API_KEY = "PUT YOUR API KEY HERE"
MODEL = "claude-sonnet-4-20250514"

CSV_IN = "ESGscopus.csv"
CSV_OUT = "topic_extractions.csv"

DEFAULT_BATCH_LIMIT = 1014
RATE_LIMIT_SLEEP = 1.2

# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful literature-review assistant.\n"
    "RESPOND IN VALID JSON ONLY. Do not wrap your answer in markdown."
)

USER_TEMPLATE = """Paper ID: {paper_id}
Title: {title}
Abstract: {abstract}

Tasks
1. For each topic below, determine whether it is meaningfully discussed. If yes, return a structured analysis with the following elements (each max 40 words):
   - problem
   - method
   - result
   - equipment (if any mentioned; else use "none")

   Topics
   a. switchgear OR switch gear
   b. panelboard OR panel board
   c. switch board OR switchboard

2. Return a JSON object with exactly this shape:
{{
  "paper_id": "<paper_id>",
  "switchgear": {{
    "presence": "yes|no",
    "problem": "<text or empty>",
    "method": "<text or empty>",
    "result": "<text or empty>",
    "equipment": "<text or 'none'>"
  }},
  "panelboard": {{
    "presence": "yes|no",
    "problem": "<text or empty>",
    "method": "<text or empty>",
    "result": "<text or empty>",
    "equipment": "<text or 'none'>"
  }},
  "switchboard": {{
    "presence": "yes|no",
    "problem": "<text or empty>",
    "method": "<text or empty>",
    "result": "<text or empty>",
    "equipment": "<text or 'none'>"
  }}
}}"""

client = anthropic.Anthropic(api_key=API_KEY)

# ---------------------------------------------------------------------
def parse_args() -> int | None:
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            value = arg.split("=", 1)[1]
            if value.lower() == "all":
                return None
            try:
                return int(value)
            except ValueError:
                pass
    return DEFAULT_BATCH_LIMIT


def call_claude(prompt: str) -> Dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


def already_processed_ids() -> Set[str]:
    ids: Set[str] = set()
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.add(row["paper_id"])
    return ids


def open_writer(mode: str):
    fieldnames = [
        "paper_id",
        "switchgear_presence", "switchgear_problem", "switchgear_method", "switchgear_result", "switchgear_equipment",
        "panelboard_presence", "panelboard_problem", "panelboard_method", "panelboard_result", "panelboard_equipment",
        "switchboard_presence", "switchboard_problem", "switchboard_method", "switchboard_result", "switchboard_equipment",
    ]
    f = open(CSV_OUT, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
    return writer, f


def main() -> None:
    batch_limit = parse_args()
    done_ids = already_processed_ids()
    processed_count = 0

    writer_mode = "a" if done_ids else "w"
    writer, outfile = open_writer(writer_mode)

    with open(CSV_IN, newline="", encoding="utf-8-sig") as src:
        reader = list(csv.DictReader(src))
        total_to_do = len([r for r in reader if r["PaperID"] not in done_ids])
        if batch_limit is not None:
            total_to_do = min(total_to_do, batch_limit)

        print(f"Total papers remaining: {total_to_do}")

        for row in reader:
            pid = row["PaperID"]
            if pid in done_ids:
                continue
            if batch_limit is not None and processed_count >= batch_limit:
                break

            print(f"[{processed_count + 1}/{total_to_do}] Processing Paper ID {pid}...")

            prompt = USER_TEMPLATE.format(
                paper_id=pid,
                title=row["Article Title"],
                abstract=row["Abstract"],
            )

            try:
                data = call_claude(prompt)
            except Exception as e:
                print(f"  Error on Paper ID {pid}: {e}")
                break

            out_row = {
                "paper_id": data["paper_id"],
            }
            for topic in ["switchgear", "panelboard", "switchboard"]:
                section = data[topic]
                out_row[f"{topic}_presence"] = section["presence"]
                out_row[f"{topic}_problem"] = section["problem"]
                out_row[f"{topic}_method"] = section["method"]
                out_row[f"{topic}_result"] = section["result"]
                out_row[f"{topic}_equipment"] = section["equipment"]

            writer.writerow(out_row)
            outfile.flush()
            processed_count += 1
            time.sleep(RATE_LIMIT_SLEEP)

    outfile.close()
    print("Run complete.")


if __name__ == "__main__":
    main()
