import csv, argparse, os
from collections import defaultdict, Counter
from statistics import mean

def load_scores(path):
    # 读取每题的5轮记录，返回 {qid: {"acc":..., "mean_p":..., "stability":..., "answers":[...]} }
    per_q = defaultdict(lambda: {"corrects":[], "ps":[], "answers":[]})
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = row.get("qid") or row.get("question_id")
            if not qid: continue
            # correct
            c = row.get("correct")
            try:
                c = int(c)
            except:
                c = 1 if str(c).lower() in ("1","true","yes") else 0
            # prob
            p = row.get("p_chosen") or row.get("prob") or row.get("confidence")
            try:
                p = float(p)
            except:
                p = None
            # answer token
            a = row.get("chosen") or row.get("pred") or row.get("answer")
            per_q[qid]["corrects"].append(c)
            if p is not None:
                per_q[qid]["ps"].append(p)
            if a is not None:
                per_q[qid]["answers"].append(a)
    out = {}
    for qid, d in per_q.items():
        acc = mean(d["corrects"]) if d["corrects"] else 0.0
        mp  = mean(d["ps"]) if d["ps"] else 0.0
        stab = len(set(d["answers"])) if d["answers"] else 0
        out[qid] = {"acc": round(acc,4), "mean_p": round(mp,4), "stability": stab}
    return out

def join_keys(*dicts):
    s=set()
    for d in dicts: s.update(d.keys())
    return sorted(s)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini",  default="outputs/race_gpt4omini_600x5/scores.csv")
    ap.add_argument("--qwen",  default="outputs/race_qwen3next80b_600x5/scores.csv")
    ap.add_argument("--deep",  default="outputs/race_deepseekv3_600x5/scores.csv")
    ap.add_argument("--out_csv", default="analysis/race_stage3_per_question.csv")
    ap.add_argument("--out_sepA", default="analysis/race_stage3_separators_A.json")
    ap.add_argument("--out_sepB", default="analysis/race_stage3_separators_B.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    m = load_scores(args.mini)
    q = load_scores(args.qwen)
    d = load_scores(args.deep)

    keys = join_keys(m,q,d)

    # 写合并表
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        hdr = [
            "qid",
            "acc_mini","acc_qwen","acc_deep",
            "p_mini","p_qwen","p_deep",
            "stab_mini","stab_qwen","stab_deep",
            "acc_range","acc_argmax","acc_argmin"
        ]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for k in keys:
            am, aq, ad = m.get(k,{}), q.get(k,{}), d.get(k,{})
            accs = {
                "mini": am.get("acc",0.0),
                "qwen": aq.get("acc",0.0),
                "deep": ad.get("acc",0.0)
            }
            # 取极值与差
            argmax = max(accs, key=lambda x: accs[x])
            argmin = min(accs, key=lambda x: accs[x])
            acc_range = round(accs[argmax]-accs[argmin],4)
            row = {
                "qid": k,
                "acc_mini": accs["mini"], "acc_qwen": accs["qwen"], "acc_deep": accs["deep"],
                "p_mini": am.get("mean_p",0.0), "p_qwen": aq.get("mean_p",0.0), "p_deep": ad.get("mean_p",0.0),
                "stab_mini": am.get("stability",0), "stab_qwen": aq.get("stability",0), "stab_deep": ad.get("stability",0),
                "acc_range": acc_range, "acc_argmax": argmax, "acc_argmin": argmin
            }
            w.writerow(row)

    # 产出分离题清单
    import json
    sepA = []  # 强分离 A：某模型≥0.8 & 另一≤0.2
    sepB = []  # 中分离 B：max-min≥0.5
    with open(args.out_csv, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            accs = {"mini": float(row["acc_mini"]), "qwen": float(row["acc_qwen"]), "deep": float(row["acc_deep"])}
            mx = max(accs.values()); mn = min(accs.values())
            # A 条件
            if mx >= 0.8 and mn <= 0.2:
                # 记录赢家/输家
                wmod = max(accs, key=lambda x: accs[x])
                lmod = min(accs, key=lambda x: accs[x])
                sepA.append({"qid": row["qid"], "winner": wmod, "loser": lmod,
                             "accs": accs})
            # B 条件
            if (mx - mn) >= 0.5:
                sepB.append({"qid": row["qid"], "argmax": max(accs, key=lambda x: accs[x]),
                             "argmin": min(accs, key=lambda x: accs[x]), "gap": round(mx-mn,4)})

    json.dump(sepA, open(args.out_sepA,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(sepB, open(args.out_sepB,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print("✅ wrote", args.out_csv)
    print("  Strong separators A (≥0.8 vs ≤0.2):", len(sepA), "->", args.out_sepA)
    print("  Mid separators B (Δacc≥0.5):", len(sepB), "->", args.out_sepB)
