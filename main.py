# pip install gliner
from gliner import GLiNER
import csv

model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
LABELS = ["person", "school", "university", "organization", "workplace", "street address",
          "city", "zip code", "email", "phone number"] # remove or add factors if we want more/less stuff

def redact(text):
    entities = model.predict_entities(text, LABELS, threshold=0.4) # lower threshold to redact more things and vice versa
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        text = text[:e["start"]] + "[REDACTED]" + text[e["end"]:]
    return text


if __name__ == "__main__":
    with open("prompts.csv", newline="", encoding="utf-8") as fin, \
         open("sanitized_prompts.csv", "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["Prompt"] = redact(row["Prompt"])
            writer.writerow(row)