import json
import csv
from openai import OpenAI

# Insert your OpenAI API key here
client = OpenAI(api_key="PUT YOUR API KEY HERE")

# Paths
input_file = "switchgear_construction_results2425_extracted.jsonl"
output_file = "switchgear_summaries.csv"

# Summarization function
def summarize_article(article_text):
    prompt = """
    You are an expert in construction supply chain issues.
    The text below is from an article about electrical switchgear delays in construction.
    Answer ONLY in valid JSON with the following keys:
    lead_time, solutions, reasons, products_affected, consequences.

    Example:
    {
      "lead_time": "Example lead time answer",
      "solutions": "Example solutions answer",
      "reasons": "Example reasons answer",
      "products_affected": "Example products/components answer",
      "consequences": "Example consequences answer"
    }

    Article:
    """ + article_text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # Force strict JSON
        messages=[
            {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# Read all lines first so we know total count
with open(input_file, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

total = len(records)
print(f"Total articles to process: {total}")

# Open CSV in append mode so we can resume or stop anytime
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["URL", "Lead Time", "Solutions", "Reasons", "Products Affected", "Consequences"])

    for idx, record in enumerate(records, start=1):
        url = record.get("href")
        article_text = record.get("extracted_text")

        if not article_text:
            print(f"[{idx}/{total}] Skipping {url} — no extracted text")
            continue

        print(f"[{idx}/{total}] Processing: {url}")

        try:
            answers = summarize_article(article_text)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            answers = {
                "lead_time": "",
                "solutions": "",
                "reasons": "",
                "products_affected": "",
                "consequences": ""
            }

        writer.writerow([
            url,
            answers.get("lead_time", ""),
            answers.get("solutions", ""),
            answers.get("reasons", ""),
            answers.get("products_affected", ""),
            answers.get("consequences", "")
        ])

        outfile.flush()  # Save progress to disk immediately

print(f"Done. Results saved to {output_file}")
