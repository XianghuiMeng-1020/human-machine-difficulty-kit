import pandas as pd, os, argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="analysis/ednet_item_answer_counts.csv")
    ap.add_argument("--out_contents", default="analysis/ednet_pseudo_contents.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.counts)
    # 对每个 item_id 按 cnt 排序取 top1
    df = df.sort_values(["item_id", "cnt"], ascending=[True, False])
    top1 = df.groupby("item_id").head(1).reset_index(drop=True)
    top1 = top1.rename(columns={"student_answer": "correct_answer"})
    os.makedirs(os.path.dirname(args.out_contents), exist_ok=True)
    top1.to_csv(args.out_contents, index=False)
    print(f"✅ wrote {args.out_contents} rows={len(top1)}")
