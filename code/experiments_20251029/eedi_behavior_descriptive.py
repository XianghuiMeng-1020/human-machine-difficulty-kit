import argparse, pandas as pd, os

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="analysis/eedi_merged_behavior.csv")
    ap.add_argument("--out_csv", default="analysis/eedi_behavior_descriptive.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.merged)

    # 同前：定义分歧
    map_m = {"简单M":0,"中等M":1,"困难M":2}
    map_h = {"简单H_proxy":0,"中等H_proxy":1,"困难H_proxy":2}
    df["diverge"] = (df["gpt4o_M"].map(map_m) != df["H_proxy"].map(map_h)).astype(int)

    # 行为特征
    df["p_gap"] = (df["gpt4o_p"] - df["mini_p"]).abs()
    df["correct_gap"] = (df["gpt4o_correct"] != df["mini_correct"]).astype(int)
    df["any_high_conf"] = ((df["gpt4o_p"]>=0.8) | (df["mini_p"]>=0.8)).astype(int)
    df["both_low_conf"] = ((df["gpt4o_p"]<0.5) & (df["mini_p"]<0.5)).astype(int)
    df["mini_says_hard"] = (df["mini_M"]=="困难M").astype(int)

    feats = ["p_gap","correct_gap","any_high_conf","both_low_conf","mini_says_hard"]

    rows=[]
    for f in feats:
        g0 = df[df["diverge"]==0][f]
        g1 = df[df["diverge"]==1][f]
        rows.append({
            "feature": f,
            "mean_div0": g0.mean(),
            "mean_div1": g1.mean(),
            "diff": g1.mean() - g0.mean(),
            "n_div0": len(g0),
            "n_div1": len(g1),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("✅ wrote", args.out_csv)
