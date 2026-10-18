"""Shuffle all responses and hide which condition produced each one (blinded scoring queue)."""
import json, os, random, re
import pandas as pd

RUNS = "results/runs.jsonl"
QUEUE = "evaluation/score_queue.jsonl"
KEY = "evaluation/condition_key.csv"
REF = re.compile(r"(quran|qur'an|surah|ayah|verse|hadith|prophet|narrated|bukhari|muslim|\b\d{1,3}:\d{1,3}\b)", re.I)

if os.path.exists("evaluation/scores.jsonl"):
    raise SystemExit("Scoring already started. Rebuilding would break the blinding.")
os.makedirs("evaluation", exist_ok=True)
runs = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
random.seed(2026)
random.shuffle(runs)
key = []
with open(QUEUE, "w", encoding="utf-8") as f:
    for i, x in enumerate(runs, 1):
        rid = f"R{i:03d}"
        f.write(json.dumps({"review_id": rid, "scenario_id": x["scenario_id"], "message": x["message"],
                "guidance_preference": x["guidance_preference"], "response": x["response"],
                "refs": sorted({m.group(0).lower() for m in REF.finditer(x["response"])})},
                ensure_ascii=False) + "\n")
        key.append({"review_id": rid, "scenario_id": x["scenario_id"],
                    "condition": x["condition"], "category": x["category"]})
pd.DataFrame(key).to_csv(KEY, index=False)
print("queue built:", len(runs), "responses. Do not open", KEY, "until scoring is done.")
