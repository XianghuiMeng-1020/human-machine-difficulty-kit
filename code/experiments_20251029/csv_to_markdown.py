import csv, argparse, os

def r4(x):
    try:
        return f"{float(x):.3f}"
    except:
        return x

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default="qid,set,p_g,p_m,dp,acc_g,acc_m,winner,overconfident,note")
    args=ap.parse_args()

    cols = args.cols.split(",")
    rows=[]
    with open(args.csv, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            # 格式化数值
            row["p_g"]=r4(row.get("p_g",""))
            row["p_m"]=r4(row.get("p_m",""))
            row["dp"]=r4(row.get("dp",""))
            row["acc_g"]=row.get("acc_g","")
            row["acc_m"]=row.get("acc_m","")
            rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # 写 markdown
    with open(args.out, "w", encoding="utf-8") as g:
        # header
        g.write("| " + " | ".join(cols) + " |\n")
        g.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for row in rows:
            g.write("| " + " | ".join(str(row.get(c,"")) for c in cols) + " |\n")

    print("✅ wrote", args.out)
