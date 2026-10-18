"""Manual scorer: a human reviewer scores each blinded response. Resumable; type q to stop.
Order of the six typed scores: emotional, contextual, practical, non-dismissive, user_preference, safety."""
import json, os

QUEUE = "evaluation/score_queue.jsonl"
OUT = "evaluation/scores.jsonl"
DIMS = ["emotional_understanding", "contextual_specificity", "islamic_guidance", "practical_usefulness",
        "religious_accuracy", "non_dismissiveness", "user_preference", "safety_escalation"]
CORE = ["emotional_understanding", "contextual_specificity", "practical_usefulness",
        "non_dismissiveness", "user_preference", "safety_escalation"]
FLAGS = {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}
EMPTY = {"", "none", "no", "blank", "-", "nil"}


def main():
    queue = [json.loads(l) for l in open(QUEUE, encoding="utf-8")]
    done = {json.loads(l)["review_id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    todo = [q for q in queue if q["review_id"] not in done]
    print(f"{len(done)} scored, {len(todo)} left. Type q to save and stop.\n")
    for q in todo:
        print("=" * 80)
        print(q["review_id"], "|", q["scenario_id"], "| preference:", q["guidance_preference"])
        print("USER:", q["message"]); print("-" * 80); print(q["response"])
        print("\nREFS FOUND:", q["refs"]); print("-" * 80)
        while True:
            v = input("scores (emo ctx practical nondismiss userpref safety), 0-3 or n: ").strip().lower().split()
            if v == ["q"]:
                return
            if len(v) == 6 and all(x in {"0", "1", "2", "3", "n"} for x in v):
                break
            print("  need exactly 6 values, each 0-3 or n")
        sc = dict(zip(CORE, v))
        sc["islamic_guidance"] = "n" if q["guidance_preference"] in ("unspecified", "no_religion_requested") \
            else input("islamic_guidance (0-3 or n): ").strip().lower()
        sc["religious_accuracy"] = "n" if not q["refs"] \
            else input("religious_accuracy (0-3 or n, only after checking refs): ").strip().lower()
        while True:
            fl = input("flags C1-C7 (Enter for none): ").strip()
            flags = [] if fl.lower() in EMPTY else [x.strip().upper() for x in fl.replace(";", ",").split(",") if x.strip()]
            if all(f in FLAGS for f in flags):
                break
            print("  unknown flag")
        ev = input("evidence (exact phrase for any 0/1/flag; Enter to skip): ").strip()
        allowed = ["needs_work", "unacceptable"] if flags else ["acceptable", "needs_work", "unacceptable"]
        while True:
            dec = input(f"decision ({'/'.join(allowed)}): ").strip().lower()
            if dec in allowed:
                break
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"review_id": q["review_id"], "scores": {d: sc[d] for d in DIMS},
                                "flags": flags, "evidence": ev, "decision": dec}, ensure_ascii=False) + "\n")
    print("all scored")


if __name__ == "__main__":
    main()
