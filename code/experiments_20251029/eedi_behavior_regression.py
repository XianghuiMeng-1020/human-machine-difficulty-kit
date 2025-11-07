import argparse, pandas as pd, numpy as np, statsmodels.api as sm, os

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--merged", default="analysis/eedi_merged_behavior.csv")
    ap.add_argument("--out", default="analysis/eedi_behavior_regression.txt")
    args=ap.parse_args()

    df = pd.read_csv(args.merged)

    # 目标：什么时候 “模型难度 != proxy 难度”
    # 先定义：模型难度 = gpt4o_M
    df["model_easy"] = (df["gpt4o_M"]=="简单M").astype(int)
    df["model_mid"]  = (df["gpt4o_M"]=="中等M").astype(int)
    df["model_hard"] = (df["gpt4o_M"]=="困难M").astype(int)

    df["proxy_easy"] = (df["H_proxy"]=="简单H_proxy").astype(int)
    df["proxy_mid"]  = (df["H_proxy"]=="中等H_proxy").astype(int)
    df["proxy_hard"] = (df["H_proxy"]=="困难H_proxy").astype(int)

    # 分歧标签：1 = 模型和 proxy 落在不同档
    df["diverge"] = (df["gpt4o_M"].map({"简单M":0,"中等M":1,"困难M":2}) !=
                     df["H_proxy"].map({"简单H_proxy":0,"中等H_proxy":1,"困难H_proxy":2})).astype(int)

    # 构造行为特征
    df["p_gap"] = (df["gpt4o_p"] - df["mini_p"]).abs()
    df["correct_gap"] = (df["gpt4o_correct"] != df["mini_correct"]).astype(int)
    df["any_high_conf"] = ((df["gpt4o_p"]>=0.8) | (df["mini_p"]>=0.8)).astype(int)
    df["both_low_conf"] = ((df["gpt4o_p"]<0.5) & (df["mini_p"]<0.5)).astype(int)
    df["mini_says_hard"] = (df["mini_M"]=="困难M").astype(int)

    feats = ["p_gap","correct_gap","any_high_conf","both_low_conf","mini_says_hard"]
    X = df[feats]
    X = sm.add_constant(X)
    y = df["diverge"]

    model = sm.Logit(y, X).fit(disp=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(model.summary().as_text())
    print("✅ wrote", args.out)
