# pip install gliner
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
LABELS = ["person", "school", "university", "organization", "workplace", "street address",
          "city", "zip code", "email", "phone number"] # remove or add factors if we want more/less stuff

def redact(text):
    entities = model.predict_entities(text, LABELS, threshold=0.4) # lower threshold to redact more things
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        text = text[:e["start"]] + "[REDACTED]" + text[e["end"]:]
    return text


if __name__ == "__main__": # can edit this to loop through a CSV
    with open("input.txt", encoding="utf-8") as f:
        text = f.read()

    result = redact(text)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)