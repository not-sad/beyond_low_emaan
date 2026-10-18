"""Make the three figures from results/scored_results.csv."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = ["emotional_understanding", "contextual_specificity", "islamic_guidance", "practical_usefulness",
        "religious_accuracy", "non_dismissiveness", "user_preference", "safety_escalation"]
os.makedirs("figures", exist_ok=True)
df = pd.read_csv("results/scored_results.csv")

means = df.groupby("condition")[DIMS].mean().T
ax = means.plot(kind="barh", figsize=(9, 6)); ax.set_xlim(0, 3); ax.set_xlabel("mean score (0-3)")
ax.set_title("Mean rubric score by dimension and condition")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig("figures/scores_by_condition.png", dpi=150); plt.close()

pd.crosstab(df.condition, df.decision).plot(kind="bar", figsize=(6, 4))
plt.title("Decisions by condition"); plt.tight_layout()
plt.savefig("figures/decisions_by_condition.png", dpi=150); plt.close()

df.pivot_table(index="guidance_preference", columns="condition", values="user_preference",
               aggfunc="mean").plot(kind="bar", figsize=(7, 4), ylim=(0, 3))
plt.title("user_preference by what the user asked"); plt.tight_layout()
plt.savefig("figures/user_preference.png", dpi=150); plt.close()
print("figures saved")
