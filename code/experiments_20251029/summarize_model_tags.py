import argparse, csv, json, os
from collections import Counter

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--tags_csv", required=True)
    ap.add_argument("--out_prefix", required=True)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    rows=[]
    with open(args.tags_csv, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)

    cnt=Counter([r["M_tau"] for r in rows])
    print("Dist:", dict(cnt), "N=", len(rows))

    easy=[r["qid"] for r in rows if r["M_tau"]=="简单M"]
    hard=[r["qid"] for r in rows if r["M_tau"]=="困难M"]
    mid=[r["qid"] for r in rows if r["M_tau"]=="中等M"]

    json.dump(easy, open(args.out_prefix+"_easy.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(hard, open(args.out_prefix+"_hard.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(mid,  open(args.out_prefix+"_mid.json","w",encoding="utf-8"),  ensure_ascii=False, indent=2)

    print("✓ easy:", len(easy), "→", args.out_prefix+"_easy.json")
    print("✓ hard:", len(hard), "→", args.out_prefix+"_hard.json")
    print("✓ mid:",  len(mid),  "→", args.out_prefix+"_mid.json")
