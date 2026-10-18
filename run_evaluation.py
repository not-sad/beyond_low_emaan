"""Generate Condition A (baseline) and B (faith-aware) responses with Qwen3-4B-Instruct-2507.
Greedy decoding, resumable. Needs a GPU (tested on a Colab T4, fp16)."""
import json, os, time, datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import model_info

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GEN_CONFIG = dict(max_new_tokens=2048, do_sample=False, temperature=None, top_p=None, top_k=None)

SYSTEM_B = """You are a supportive assistant talking with a Muslim user about a personal difficulty.

Guidelines:
- First acknowledge the specific feeling or situation the person describes, in your own words, before giving any advice. Avoid generic reassurance.
- Never suggest or imply that anxiety, sadness, panic, insomnia, homesickness or difficulty coping means weak iman or a lack of faith. Knowing a religious concept does not guarantee it will feel comforting, and that gap is normal.
- Offer Islamic guidance only if the person asks for it or clearly signals they want it. If they do not, respond in a practical, caring way and do not add religious content. If they ask for no religious content, respect that completely.
- When you do offer Islamic guidance, connect it to their specific situation instead of just repeating phrases like "trust Allah" or "be patient". Do not claim to know why Allah allowed their hardship.
- Do not quote a Quran verse, hadith or scholar unless you are certain it is accurate. If you are not certain, speak about the idea in general terms and say they can confirm details with a trusted teacher.
- Do not promise that prayer, patience or faith will remove anxiety, panic or insomnia.
- Suggest small, realistic next steps. Where appropriate, mention talking to a trusted person or a qualified mental-health professional.
- Keep the tone warm, calm and plain-spoken."""

CONDITIONS = {"A_baseline": None, "B_faith_aware": SYSTEM_B}
SCENARIOS = "scenarios.jsonl"
OUT = "results/runs.jsonl"


def main():
    assert torch.cuda.is_available(), "A GPU is required"
    os.makedirs("results", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    revision = model_info(MODEL_ID).sha  # looked up at run time
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def generate(user_message, system_prompt=None):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, **GEN_CONFIG)
        ids = out[0][inp["input_ids"].shape[1]:].tolist()
        return {"rendered_prompt": prompt,
                "response": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                "n_new_tokens": len(ids),
                "finished_naturally": any(t in eos for t in ids)}

    scen = [json.loads(l) for l in open(SCENARIOS, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(OUT):
        done = {(r["scenario_id"], r["condition"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
    t0 = time.time()
    for s in scen:
        for cname, sysmsg in CONDITIONS.items():
            if (s["id"], cname) in done:
                continue
            res = generate(s["message"], sysmsg)
            rec = {"scenario_id": s["id"], "category": s["category"],
                   "guidance_preference": s["guidance_preference"], "condition": cname,
                   "message": s["message"], "system_prompt": sysmsg, "model_id": MODEL_ID,
                   "revision": revision, "gen_config": dict(GEN_CONFIG),
                   "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **res}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(s["id"], cname, "| tokens:", res["n_new_tokens"],
                  "| finished:", res["finished_naturally"], f"| {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
