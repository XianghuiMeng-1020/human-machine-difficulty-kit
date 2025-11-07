import pandas as pd, os

glob_path = "analysis/global/global_alignment_table.csv"
full_path = "analysis/ednet_proxy_labels_full.csv"

glob = pd.read_csv(glob_path)
full = pd.read_csv(full_path)

dist = full["H_proxy"].value_counts(normalize=True).to_dict()
p_easy = dist.get("简单H_proxy", 0.0)
p_mid  = dist.get("中等H_proxy", 0.0)
p_hard = dist.get("困难H_proxy", 0.0)
p_low  = dist.get("低曝光", 0.0)

row = {
    "dataset": "EdNet-KT1",
    "variant": "full-95M-proxy",
    "model": "-",
    # 我们这里用 “(1 - 低曝光) * (easy+mid)” 做个 proxy alignment，占位
    "alignment": float(1.0 - p_low),
    "notes": f"easy={p_easy:.3f}, mid={p_mid:.3f}, hard={p_hard:.3f}, low={p_low:.5f}",
}

glob = pd.concat([glob, pd.DataFrame([row])], ignore_index=True)
glob.to_csv(glob_path, index=False)
print("✅ updated", glob_path)
print(row)
