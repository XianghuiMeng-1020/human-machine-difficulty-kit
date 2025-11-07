import os, argparse, pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="analysis/ednet_flat_ednet_true.csv")
    ap.add_argument("--pseudo_contents", default="analysis/ednet_pseudo_contents.csv")
    ap.add_argument("--out", default="analysis/ednet_flat_with_correct.csv")
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    cont = pd.read_csv(args.pseudo_contents)
    ans_map = dict(zip(cont["item_id"].astype(str), cont["correct_answer"].astype(str)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 为了能一点点写，我们先把文件头写进去
    out_path = args.out
    first = True
    total = 0

    for chunk in pd.read_csv(args.logs, chunksize=args.chunksize):
        chunk["item_id"] = chunk["item_id"].astype(str)
        chunk["student_answer"] = chunk["student_answer"].astype(str)

        gold = chunk["item_id"].map(ans_map)
        chunk["gold_answer"] = gold
        chunk["correct"] = (chunk["student_answer"] == gold).astype("Int64")

        mode = "w" if first else "a"
        header = first
        chunk.to_csv(out_path, mode=mode, header=header, index=False)
        first = False

        total += len(chunk)
        print(f"[apply] wrote {total} rows ...")

    print(f"✅ wrote {out_path}")
