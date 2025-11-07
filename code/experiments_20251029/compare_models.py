import csv, argparse, os, json
from collections import defaultdict, Counter

def load_tags(path):
    d={}
    with open(path, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            qid=row.get("qid") or row.get("question_id")
            if not qid: continue
            d[str(qid)]=row["M_tau"]
    return d

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpt4o", required=True)
    ap.add_argument("--mini", required=True)
    ap.add_argument("--out_prefix", default="analysis/model_vs_model_tau08")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    a=load_tags(args.gpt4o)
    b=load_tags(args.mini)
    common=sorted(set(a)&set(b))

    # 3×3 confusion
    cols=["简单M","中等M","困难M"]
    mat=defaultdict(lambda: Counter())
    for q in common:
        mat[a[q]][b[q]] += 1

    # 保存矩阵
    cm_path=args.out_prefix+"_confusion.csv"
    with open(cm_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["gpt4o \\ mini"]+cols)
        for row in cols:
            w.writerow([row]+[mat[row][c] for c in cols])

    # 分歧清单
    mini_hard_gpt_not = [q for q in common if b[q]=="困难M" and a[q] in {"简单M","中等M"}]
    mini_easy_gpt_not = [q for q in common if b[q]=="简单M" and a[q] in {"困难M","中等M"}]

    json.dump(mini_hard_gpt_not, open(args.out_prefix+"_miniHard_gptNot.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(mini_easy_gpt_not, open(args.out_prefix+"_miniEasy_gptNot.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print("N common:", len(common))
    print("Saved confusion:", cm_path)
    print("mini=困难M & gpt4o∈{简单/中等} :", len(mini_hard_gpt_not), "→", args.out_prefix+"_miniHard_gptNot.json")
    print("mini=简单M & gpt4o∈{困难/中等} :", len(mini_easy_gpt_not), "→", args.out_prefix+"_miniEasy_gptNot.json")
