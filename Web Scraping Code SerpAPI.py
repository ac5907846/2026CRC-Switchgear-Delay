import os
import csv
import time
import requests
from tqdm import tqdm
from newspaper import Article

# ========== CONFIGURATION ==========

SERPAPI_KEY = "PUT YOUR API KEY HERE"

YEARS = list(range(2021, 2025))
KEYWORDS = [
    "electrical switchgear delays construction",
    "switchgear supply chain disruption",
    "construction project delays due to electrical equipment",
    "electrical switchgear procurement challenges",
    "construction schedule delays switchgear",
    "supply chain issues electrical switchgear",
    "switchgear lead time increase",
    "electrical gear backlog construction",
    "construction procurement delay electrical components",
    "contractor delays electrical switchgear",
    "global supply chain impact on switchgear",
    "construction industry electrical equipment shortage",
    "building project delays due to switchgear",
    "switchgear shortage infrastructure project",
    "COVID-19 switchgear supply chain disruption",
    "pandemic construction equipment delays",
    "switchgear delivery time issues",
    "construction site electrical supply chain risks",
    "EPC delays switchgear supply chain",
    "tariff effects switchgear construction procurement",
    "inflation and electrical equipment procurement",
    "electrical switchgear shipping delays",
    "electrical equipment availability construction sector",
    "global trade delays in construction materials",
    "switchgear logistics disruptions construction",
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

OUTPUT_DIR = os.path.dirname(__file__)  # save next to this script
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "switchgear_construction_results2425.csv")
VIEW_COPY_FILE = os.path.join(OUTPUT_DIR, "switchgear_construction_results_view.csv")  # optional

FIELDNAMES = ["Year", "Keyword", "Result Title", "Link", "Extracted Title", "Summary"]

SAVE_EVERY_N_ROWS = 25  # how often to refresh the view copy

# ========== IO HELPERS ==========

def ensure_csv_with_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def load_existing_links(path):
    links = set()
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                link = row.get("Link")
                if link:
                    links.add(link)
    return links

def append_row(row):
    # append and flush each row so you can watch the file grow
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())  # force write to disk

def refresh_view_copy(rows_written_since_copy):
    # optional - keeps a fresh copy you can open in Excel without locking the main file
    if rows_written_since_copy >= SAVE_EVERY_N_ROWS:
        try:
            with open(OUTPUT_FILE, "rb") as src, open(VIEW_COPY_FILE, "wb") as dst:
                dst.write(src.read())
            return 0
        except Exception:
            # ignore copy errors - continue scraping
            return rows_written_since_copy
    return rows_written_since_copy

# ========== SEARCH AND SCRAPE ==========

def search_google(query, year):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "en",
        "num": 10,
        "tbs": f"cdr:1,cd_min:1/1/{year},cd_max:12/31/{year}",
    }
    r = requests.get("https://serpapi.com/search", params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return [], f"HTTP {r.status_code}"
    data = r.json()
    # SerpAPI can return an 'error' field when over the limit
    if "error" in data:
        return [], data["error"]
    return data.get("organic_results", []), None

def summarize_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        # nlp() sometimes fails if the text is tiny - wrap it
        try:
            article.nlp()
            summary = article.summary
        except Exception:
            summary = article.text[:1000]  # fallback - first 1000 chars
        return article.title or "", summary or ""
    except Exception:
        return "", ""

# ========== MAIN ==========

def main():
    ensure_csv_with_header(OUTPUT_FILE)
    processed_links = load_existing_links(OUTPUT_FILE)
    print(f"Loaded {len(processed_links)} existing rows - will skip duplicates")

    rows_since_copy = 0
    total_new = 0

    try:
        for year in YEARS:
            for keyword in KEYWORDS:
                print(f"\nSearching '{keyword}' in {year}")
                results, err = search_google(keyword, year)
                if err:
                    print(f"SerpAPI issue: {err}")
                    # If over limit - stop cleanly so you keep what you have
                    if "limit" in err.lower() or "exceeded" in err.lower():
                        print("API limit hit - stopping. You can rerun later and it will resume on the same CSV.")
                        return

                for result in tqdm(results):
                    url = result.get("link")
                    if not url or url in processed_links:
                        continue

                    title, summary = summarize_article(url)
                    if not title and not summary:
                        continue

                    row = {
                        "Year": year,
                        "Keyword": keyword,
                        "Result Title": result.get("title", ""),
                        "Link": url,
                        "Extracted Title": title,
                        "Summary": summary,
                    }
                    append_row(row)
                    processed_links.add(url)
                    total_new += 1
                    rows_since_copy += 1
                    # be nice to sites and your API quota
                    time.sleep(1)

                    rows_since_copy = refresh_view_copy(rows_since_copy)

    except KeyboardInterrupt:
        print("\nStopped by user - all progress has been saved to CSV.")
    finally:
        # final view copy
        try:
            refresh_view_copy(SAVE_EVERY_N_ROWS)
        except Exception:
            pass

    print(f"\nDone. Added {total_new} new rows. File: {OUTPUT_FILE}")
    print(f"If Excel needs a live view copy, open: {VIEW_COPY_FILE}")

if __name__ == "__main__":
    main()
