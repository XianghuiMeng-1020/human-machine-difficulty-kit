import os
import pandas as pd

rows = []

# 1) Eedi sparse
eedi_sparse = "analysis/eedi_all_alignment.csv"
if os.path.exists(eedi_sparse):
    df = pd.read_csv(eedi_sparse)
    for _, r in df.iterrows():
        rows.append({
            "dataset": "Eedi",
            "variant": "human-sparse",
            "model": r.get("folder","-"),
            "alignment": r.get("alignment", r.get("align", None)),
            "notes": "raw sparse cross (most MissingH)",
        })

def load_proxy_x_model(path, dataset, model, note):
    """try several formats."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # case 1: already has total/aligned
    if "total" in df.columns and "aligned" in df.columns:
        tot = int(df["total"].iloc[0])
        ali = int(df["aligned"].iloc[0])
        return {
            "dataset": dataset,
            "variant": "proxy-dense",
            "model": model,
            "alignment": ali / tot if tot else 0.0,
            "notes": note,
        }
    # case 2: pivot-like: first col is proxy(H), other cols are model(M)
    # we just take diagonal / sum(all)
    cols = list(df.columns)
    hcol = cols[0]
    total = 0
    diag = 0
    for _, row in df.iterrows():
        h_label = str(row[hcol]).strip()
        for mcol in cols[1:]:
            v = int(row[mcol])
            total += v
            if h_label in mcol:  # e.g. 简单H vs 简单M
                diag += v
    return {
        "dataset": dataset,
        "variant": "proxy-dense",
        "model": model,
        "alignment": diag / total if total else 0.0,
        "notes": note + " (pivot)",
    }

# 2) Eedi proxy × model
r1 = load_proxy_x_model(
    "analysis/eedi_proxy_x_model_gpt4o_tau08.csv",
    "Eedi",
    "gpt4o",
    "proxy vs model @τ=0.8",
)
if r1: rows.append(r1)

r2 = load_proxy_x_model(
    "analysis/eedi_proxy_x_model_gpt4omini_tau08.csv",
    "Eedi",
    "gpt4o-mini",
    "proxy vs model @τ=0.8",
)
if r2: rows.append(r2)

# 3) RACE proxy 600×5
race_align = "paper_assets/mv-hmda_race/stage3_alignment_summary_race.csv"
if os.path.exists(race_align):
    df = pd.read_csv(race_align)
    for _, r in df.iterrows():
        rows.append({
            "dataset": "RACE",
            "variant": "proxy-5runs",
            "model": r["model"],
            "alignment": r["alignment"],
            "notes": f"n={int(r['n'])}",
        })

# 4) EdNet scaling
ednet_scale = "analysis/ednet_scale/ednet_scaling.csv"
if os.path.exists(ednet_scale):
    df = pd.read_csv(ednet_scale)
    for _, r in df.iterrows():
        rows.append({
            "dataset": "EdNet-KT1",
            "variant": f"slice-{r['slice']}",
            "model": "-",
            # 用困难占比当作“可恢复的人难密度”
            "alignment": r["p_hard"],
            "notes": f"users={int(r['n_users'])} low={r['p_low']:.2f}",
        })

out = pd.DataFrame(rows)
os.makedirs("analysis/global", exist_ok=True)
out.to_csv("analysis/global/global_alignment_table.csv", index=False)
print("✅ wrote analysis/global/global_alignment_table.csv")
print(out.to_string(index=False))
