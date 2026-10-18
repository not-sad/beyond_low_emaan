"""Unblind the scores and print summary tables. Run only after every response is scored."""
import json, os
import numpy as np
import pandas as pd

DIMS = ["emotional_understanding", "contextual_specificity", "islamic_guidance", "practical_usefulness",
        "religious_accuracy", "non_dismissiveness", "user_preference", "safety_escalation"]

key = pd.read_csv("evaluation/condition_key.csv")
rows = []
for r in map(json.loads, open("evaluation/scores.jsonl", encoding="utf-8")):
    row = {"review_id": r["review_id"], "decision": r["decision"],
           "flags": ";".join(r["flags"]), "n_flags": len(r["flags"])}
    for d in DIMS:
        row[d] = np.nan if r["scores"][d] == "n" else float(r["scores"][d])
    rows.append(row)
df = pd.DataFrame(rows).merge(key, on="review_id")
assert len(df) == len(key), f"only {len(df)} of {len(key)} scored"
pref = {json.loads(l)["id"]: json.loads(l)["guidance_preference"] for l in open("scenarios.jsonl", encoding="utf-8")}
df["guidance_preference"] = df.scenario_id.map(pref)
os.makedirs("results", exist_ok=True)
df.to_csv("results/scored_results.csv", index=False)

print("=== mean per dimension (N/A excluded) ==="); print(df.groupby("condition")[DIMS].mean().round(2).T)
print("\n=== n behind each mean ===");               print(df.groupby("condition")[DIMS].count().T)
print("\n=== decisions ===");                           print(pd.crosstab(df.condition, df.decision))
fl = df[df.n_flags > 0].assign(f=lambda d: d.flags.str.split(";")).explode("f")
print("\n=== flags by type ===");                       print(fl.groupby(["condition", "f"]).size() if len(fl) else "none")
print("\n=== user_preference by what was asked ===")
print(df.pivot_table(index="guidance_preference", columns="condition", values="user_preference",
                     aggfunc=["mean", "count"]).round(2))
df["avg_dims"] = df[DIMS].mean(axis=1)
print("\n=== category x condition ===")
print(df.pivot_table(index="category", columns="condition", values="avg_dims", aggfunc="mean").round(2))
if os.path.exists("evaluation/citations.csv"):
    print("\n=== citation check ==="); print(pd.read_csv("evaluation/citations.csv").verdict.value_counts())
