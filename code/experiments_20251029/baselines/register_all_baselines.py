import os
import pandas as pd

GLOBAL = "analysis/global/global_alignment_table.csv"


def pick_test_acc(path: str):
    """
    从常见的 baseline 输出里挑 test 集的准确率/对齐率。
    支持三种格式：
    1) 有列: split, acc  → 取 split == "test"
    2) 有列: split, accuracy → 取 split == "test"
    3) 单行里有: test_acc / acc / accuracy → 取这一列
    找不到就返回 None
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    # 情况 1/2：多行、有 split
    if "split" in df.columns:
        metric_col = None
        if "acc" in df.columns:
            metric_col = "acc"
        elif "accuracy" in df.columns:
            metric_col = "accuracy"
        if metric_col is not None:
            sub = df[df["split"].astype(str).str.lower() == "test"]
            if len(sub) > 0:
                return float(sub.iloc[0][metric_col])

    # 情况 3：单行
    for cand in ["test_acc", "acc", "accuracy"]:
        if cand in df.columns:
            return float(df.iloc[0][cand])

    return None


def main():
    rows = []

    # 1) BEA24 LLM-feat baseline
    p_bea_llm = "baselines/bea24_llm_diff/out/llm_feat_vs_mvhmda.csv"
    acc_bea_llm = pick_test_acc(p_bea_llm)
    if acc_bea_llm is not None:
        rows.append(
            {
                "dataset": "mixed(eedi/race/syn)",
                "variant": "BEA24-LLM-feat (logreg)",
                "model": "-",
                "alignment": acc_bea_llm,
                "notes": "feature-only baseline",
            }
        )

    # 2) BEA24 BOW baseline
    p_bea_bow = "baselines/bea24_llm_diff/out/bow_vs_mvhmda.csv"
    acc_bea_bow = pick_test_acc(p_bea_bow)
    if acc_bea_bow is not None:
        rows.append(
            {
                "dataset": "mixed(eedi/race/syn)",
                "variant": "BEA24-BOW (templated)",
                "model": "-",
                "alignment": acc_bea_bow,
                "notes": "content-only baseline",
            }
        )

    # 3) CL4KT stub (no torch)
    p_cl4kt_stub = "baselines/cl4kt_diff/out/cl4kt_stub_vs_mvhmda.csv"
    if os.path.exists(p_cl4kt_stub):
        df_stub = pd.read_csv(p_cl4kt_stub)
        for _, r in df_stub.iterrows():
            if "acc" not in r or pd.isna(r["acc"]):
                continue
            rows.append(
                {
                    "dataset": r["dataset"],
                    "variant": r["method"],
                    "model": "-",
                    "alignment": float(r["acc"]),
                    "notes": "KT-style baseline (no torch)",
                }
            )

    # 4) CL4KT strict/nextstep 你也跑了，可以顺手并进来
    for extra_csv in [
        "baselines/cl4kt_diff/out/cl4kt_strict_vs_mvhmda.csv",
        "baselines/cl4kt_diff/out/cl4kt_nextstep_vs_mvhmda.csv",
    ]:
        if os.path.exists(extra_csv):
            df_extra = pd.read_csv(extra_csv)
            for _, r in df_extra.iterrows():
                if "acc" not in r or pd.isna(r["acc"]):
                    continue
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "variant": r["method"],
                        "model": "-",
                        "alignment": float(r["acc"]),
                        "notes": "KT-style baseline",
                    }
                )

    # 合进 global 表
    glob = pd.read_csv(GLOBAL)
    if rows:
        glob = pd.concat([glob, pd.DataFrame(rows)], ignore_index=True)
        glob.to_csv(GLOBAL, index=False)
        print("✅ updated", GLOBAL)
    else:
        print("no baseline rows found, global not changed")


if __name__ == "__main__":
    main()
