import csv, os

PROXY = "paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv"
MODELS = {
    "gpt4omini": "paper_assets/mv-hmda_race/stage2_model_tags_race_gpt4omini_tau08.csv",
    "qwen3next80b": "paper_assets/mv-hmda_race/stage2_model_tags_race_qwen3next80b_tau08.csv",
    "deepseekv3": "paper_assets/mv-hmda_race/stage2_model_tags_race_deepseekv3_tau08.csv",
}

# 读 proxy
proxy = {}
with open(PROXY, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        proxy[row["question_id"]] = row["H_proxy"]

out_rows = []
for mid, path in MODELS.items():
    if not os.path.exists(path):
        continue
    ok = 0; tot = 0
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = row["qid"]
            mtag = row["M_tau"]
            if qid not in proxy:
                continue
            htag = proxy[qid]
            # 简单映射：简单M ↔ 简单H_proxy; 中等M ↔ 中等H_proxy; 困难M ↔ 困难H_proxy
            mapM = {"简单M":"简单H_proxy","中等M":"中等H_proxy","困难M":"困难H_proxy"}
            if mtag in mapM:
                if mapM[mtag] == htag:
                    ok += 1
                tot += 1
    align = ok / tot if tot>0 else 0.0
    out_rows.append({"model": mid, "n": tot, "alignment": round(align,4)})

OUT = "paper_assets/mv-hmda_race/stage3_alignment_summary_race.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    hdr = ["model","n","alignment"]
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out_rows)

print("✅ wrote", OUT)
for r in out_rows:
    print("  ", r)
