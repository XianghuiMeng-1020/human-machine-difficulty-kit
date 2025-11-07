import argparse, csv, os
from collections import defaultdict

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--proxy", default="analysis/eedi_proxy_labels.csv")
    ap.add_argument("--gpt4o_tags", default="analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--mini_tags",  default="analysis/eedi_gpt4omini_tau08_model_tags.csv")
    ap.add_argument("--out", default="analysis/eedi_merged_behavior.csv")
    args=ap.parse_args()

    # 1) proxy
    proxy = {}
    with open(args.proxy, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = row["question_id"]
            proxy[qid] = {
                "H_proxy": row["H_proxy"],
                "gpt4o_correct": row["gpt4o_correct"],
                "gpt4o_p": row["gpt4o_p"],
                "mini_correct": row["mini_correct"],
                "mini_p": row["mini_p"],
            }

    # 2) gpt4o tags
    gpt_tags = {}
    with open(args.gpt4o_tags, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            gpt_tags[row["qid"]] = row["M_tau"]

    # 3) mini tags
    mini_tags = {}
    with open(args.mini_tags, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mini_tags[row["qid"]] = row["M_tau"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr = [
            "qid",
            "H_proxy",
            "gpt4o_M",
            "mini_M",
            "gpt4o_correct",
            "gpt4o_p",
            "mini_correct",
            "mini_p",
        ]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for qid, info in proxy.items():
            w.writerow({
                "qid": qid,
                "H_proxy": info["H_proxy"],
                "gpt4o_M": gpt_tags.get(qid, "MissingM"),
                "mini_M": mini_tags.get(qid, "MissingM"),
                "gpt4o_correct": info["gpt4o_correct"],
                "gpt4o_p": info["gpt4o_p"],
                "mini_correct": info["mini_correct"],
                "mini_p": info["mini_p"],
            })
    print("✅ wrote", args.out)
