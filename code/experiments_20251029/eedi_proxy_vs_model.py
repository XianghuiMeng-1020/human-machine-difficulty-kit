import argparse, csv, os
from collections import Counter

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="analysis/eedi_proxy_labels.csv")
    ap.add_argument("--model_tags", default="analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--out", default="analysis/eedi_proxy_x_model_gpt4o_tau08.csv")
    args = ap.parse_args()

    # 1) 读 proxy
    proxy = {}
    with open(args.proxy, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            proxy[row["question_id"]] = row["H_proxy"]

    # 2) 读模型端标签
    rows=[]
    with open(args.model_tags, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            qid = row["qid"]
            m = row["M_tau"]
            rows.append((qid, m))

    # 3) 交叉统计
    cross = Counter()
    total = 0
    aligned = 0
    for qid, m in rows:
        h = proxy.get(qid, "Missing_proxy")
        cross[(h,m)] += 1
        total += 1
        if h=="简单H_proxy" and m=="简单M":
            aligned += 1
        elif h=="中等H_proxy" and m=="中等M":
            aligned += 1
        elif h=="困难H_proxy" and m=="困难M":
            aligned += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr=["H_proxy","简单M","中等M","困难M"]
        w=csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        # 把三个 proxy 类别都写出来
        for h in ["简单H_proxy","中等H_proxy","困难H_proxy","Missing_proxy"]:
            row = {
                "H_proxy": h,
                "简单M": cross.get((h,"简单M"), 0),
                "中等M": cross.get((h,"中等M"), 0),
                "困难M": cross.get((h,"困难M"), 0),
            }
            w.writerow(row)

    ratio = aligned / total if total>0 else 0.0
    print("total:", total)
    print("aligned:", aligned)
    print("proxy vs model diagonal ratio:", f"{ratio:.4f}")
    print("✅ wrote", args.out)
