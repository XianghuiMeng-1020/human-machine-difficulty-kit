import csv, argparse, os

def load_csv(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def pick_representative(rows, k=3):
    # 选择策略：优先 |dp| 大、且 acc 差异明显（|da|=1）
    rows2=sorted(rows, key=lambda r: (abs(float(r["dp"])), abs(int(float(r["da"])))), reverse=True)
    return rows2[:k]

def note_for(row):
    qid=row["qid"]; setn=row["set"]
    dp=float(row["dp"]); da=int(float(row["da"]))
    pg=float(row["p_g"]); pm=float(row["p_m"])
    ag=int(float(row["acc_g"])); am=int(float(row["acc_m"]))
    if setn=="miniHard_gptNot":
        # mini判难
        core = "mini 判难，{} {}；Δp={:+.2f}（gpt4o−mini）".format(
            "gpt4o更准" if ag>am else "mini更准" if am>ag else "两者同错/同对",
            "(过度自信)" if (ag==0 and pg>0.8) or (am==0 and pm>0.8) else "",
            dp
        )
    else:
        # mini判易
        core = "mini 判易，{}；Δp={:+.2f}（gpt4o−mini）".format(
            "mini更准" if am>ag else "gpt4o更准" if ag>am else "两者同错/同对",
            dp
        )
    return f"- qid {qid} [{setn}]: {core}"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard_csv", required=True)
    ap.add_argument("--easy_csv", required=True)
    ap.add_argument("--out", default="analysis/eedi_tau08_case_notes.md")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hard = load_csv(args.hard_csv)
    easy = load_csv(args.easy_csv)

    sel = pick_representative(hard, k=3) + pick_representative(easy, k=3)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Eedi τ=0.8 分歧案例速记（Auto）\n\n")
        for r in sel:
            f.write(note_for(r) + "\n")

    print("✅ wrote", args.out)
