import csv, argparse, os

def load_csv(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def label_row(r):
    acc_g = int(float(r["acc_g"]))  # 0/1
    acc_m = int(float(r["acc_m"]))
    p_g   = float(r["p_g"])
    p_m   = float(r["p_m"])
    dp    = float(r["dp"])
    da    = int(acc_g) - int(acc_m)

    # 谁更准
    if acc_g > acc_m: winner = "gpt4o"
    elif acc_g < acc_m: winner = "mini"
    else: winner = "tie"

    # 谁在“错时更自信”（overconfidence）
    over = "none"
    if acc_g==0 and acc_m==1 and p_g>0.8: over="gpt4o_overconf"
    if acc_m==0 and acc_g==1 and p_m>0.8: over="mini_overconf"
    if acc_g==0 and acc_m==0:
        over = "both_high_conf" if (p_g>0.8 or p_m>0.8) else "both_low_conf"

    # 备注
    if r["set"]=="miniHard_gptNot":
        # mini 认为难；若 gpt4o 更准 & 置信更高，提示“大模型鲁棒”
        note = "mini判难; " + ("gpt4o更准" if winner=="gpt4o" else "mini更准" if winner=="mini" else "平局")
        if dp>0.15: note += "; gpt4o更自信"
        if dp<-0.15: note += "; mini更自信"
    else:
        # mini 认为易；若 mini 更准 & 更自信，可能是“题型捷径”
        note = "mini判易; " + ("mini更准" if winner=="mini" else "gpt4o更准" if winner=="gpt4o" else "平局")
        if dp>0.15: note += "; gpt4o更自信"
        if dp<-0.15: note += "; mini更自信(或捷径)"
    return winner, over, note

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard_csv", required=True)   # analysis/eedi_tau08_deltaP_miniHard_top10.csv
    ap.add_argument("--easy_csv", required=True)   # analysis/eedi_tau08_deltaP_miniEasy_top10.csv
    ap.add_argument("--out_prefix", default="analysis/eedi_tau08_case_report")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    rows = load_csv(args.hard_csv) + load_csv(args.easy_csv)

    out_rows=[]
    for r in rows:
        winner, over, note = label_row(r)
        out_rows.append({
            "qid": r["qid"], "set": r["set"],
            "p_g": r["p_g"], "p_m": r["p_m"], "dp": r["dp"],
            "acc_g": r["acc_g"], "acc_m": r["acc_m"], "da": r["da"],
            "winner": winner, "overconfident": over, "note": note
        })

    # 导出 CSV
    out_csv = args.out_prefix + "_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        hdr=["qid","set","p_g","p_m","dp","acc_g","acc_m","da","winner","overconfident","note"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out_rows)
    print("✅ wrote", out_csv)

    # 简要计数
    from collections import Counter
    win_cnt = Counter([r["winner"] for r in out_rows])
    over_cnt= Counter([r["overconfident"] for r in out_rows])
    print("Winners:", dict(win_cnt))
    print("Overconf:", dict(over_cnt))
