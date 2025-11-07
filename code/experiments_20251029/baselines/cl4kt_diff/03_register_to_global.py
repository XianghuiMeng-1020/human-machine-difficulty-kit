import os, pandas as pd

glob_path = "analysis/global/global_alignment_table.csv"
glob = pd.read_csv(glob_path)

stub_path = "baselines/cl4kt_diff/out/cl4kt_stub_vs_mvhmda.csv"
if os.path.exists(stub_path):
    stub = pd.read_csv(stub_path)
    rows = []
    for _, r in stub.iterrows():
        if pd.isna(r["acc"]):
            continue
        rows.append({
            "dataset": r["dataset"],
            "variant": r["method"],
            "model": "-",
            "alignment": float(r["acc"]),
            "notes": "KT-style baseline (no torch)",
        })
    glob = pd.concat([glob, pd.DataFrame(rows)], ignore_index=True)

os.makedirs("analysis/global", exist_ok=True)
glob.to_csv(glob_path, index=False)
print("✅ updated", glob_path)
