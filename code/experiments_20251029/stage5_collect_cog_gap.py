import json, os

OUT = "paper_assets/mv-hmda/stage5_cognitive_gap_items.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def dump_items(items, source, f):
    for q in items:
        obj = {
            "source": source,
            "qid": q if isinstance(q, str) else q.get("qid",""),
            "payload": q,
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(OUT, "w", encoding="utf-8") as f:
    # 1) Eedi: mini 说难但 gpt4o 不难
    try:
        eedi_mini_hard = json.load(open("analysis/eedi_tau08_gpt4o_vs_mini_miniHard_gptNot.json", encoding="utf-8"))
        dump_items(eedi_mini_hard, "eedi_miniHard_gptNot", f)
    except FileNotFoundError:
        pass

    # 2) RACE: 强分离 A
    try:
        race_A = json.load(open("analysis/race_stage3_separators_A.json", encoding="utf-8"))
        dump_items(race_A, "race_sepA", f)
    except FileNotFoundError:
        pass

    # 3) synthetic real
    try:
        synth = json.load(open("analysis/synthetic_real/synthetic_separators.json", encoding="utf-8"))
        dump_items(synth, "synthetic_real", f)
    except FileNotFoundError:
        pass

print("✅ wrote", OUT)
