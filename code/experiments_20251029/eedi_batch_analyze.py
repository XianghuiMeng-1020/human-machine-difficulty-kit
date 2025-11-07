import os, csv, argparse

def find_eedi_dirs(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 我们只关心里面同时有 scores.csv 和 human_x_model_tau080.csv 的目录
        if "scores.csv" in filenames and "human_x_model_tau080.csv" in filenames:
            if "eedi" in dirpath:
                out.append(dirpath)
    return sorted(list(set(out)))

def to_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def load_cross(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            h = row.get("\ufeffH_label") or row.get("H_label") or row.get("H") or row.get("label")
            rows.append({
                "H": h,
                "M_easy": to_int(row.get("简单M", 0)),
                "M_mid":  to_int(row.get("中等M", 0)),
                "M_hard": to_int(row.get("困难M", 0)),
            })
    return rows

def alignment_from_cross(rows):
    total = 0
    aligned = 0
    for r in rows:
        csum = r["M_easy"] + r["M_mid"] + r["M_hard"]
        total += csum
        if r["H"] == "简单H":
            aligned += r["M_easy"]
        elif r["H"] == "中等H":
            aligned += r["M_mid"]
        elif r["H"] == "困难H":
            aligned += r["M_hard"]
        else:
            # MissingH 之类的就不记到 aligned 里，但 total 里还是要算
            pass
    if total == 0:
        return 0.0, 0
    return aligned / total, total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="analysis/eedi_all_alignment.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dirs = find_eedi_dirs(args.root)
    print("found eedi dirs:", len(dirs))

    out_rows = []
    for d in dirs:
        cross80 = os.path.join(d, "human_x_model_tau080.csv")
        cross90 = os.path.join(d, "human_x_model_tau090.csv")

        rows80 = load_cross(cross80)
        a80, n80 = alignment_from_cross(rows80)

        if os.path.exists(cross90):
            rows90 = load_cross(cross90)
            a90, n90 = alignment_from_cross(rows90)
        else:
            a90, n90 = "", ""

        out_rows.append({
            "dir": d,
            "align_tau080": f"{a80:.4f}",
            "n_tau080": n80,
            "align_tau090": f"{a90 if a90=='' else f'{a90:.4f}'}",
            "n_tau090": n90,
        })
        print(f"[{d}] τ=0.80 align={a80:.4f} (n={n80})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr = ["dir", "align_tau080", "n_tau080", "align_tau090", "n_tau090"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(out_rows)

    print("✅ wrote", args.out)
