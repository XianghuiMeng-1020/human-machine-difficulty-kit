import argparse, csv

def load_matrix(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    cols = rows[0][1:]
    data = {rows[i][0]: {cols[j]: int(rows[i][1:][j]) for j in range(len(cols))}
            for i in range(1, len(rows))}
    return data, cols

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", required=True)
    ap.add_argument("--exclude", default="MissingH", help="comma-separated human rows to exclude")
    ap.add_argument("--map", default="简单H=简单M,中等H=中等M,困难H=困难M")
    args = ap.parse_args()

    table, cols = load_matrix(args.cross)
    exclude = set([s.strip() for s in args.exclude.split(",") if s.strip()])
    mapping = dict(pair.split("=") for pair in args.map.split(","))
    # 过滤
    table = {h: row for h, row in table.items() if h not in exclude}

    N = sum(sum(row.values()) for row in table.values())
    diag = 0
    per_diag = {}
    for h, row in table.items():
        m = mapping.get(h)
        v = row.get(m, 0)
        diag += v
        per_diag[h] = v

    print(f"File: {args.cross}")
    print(f"N_labeled = {N}")
    print(f"Aligned (diagonal) = {diag}")
    print(f"Alignment (labeled-only) = {diag / N if N else 0:.4f}")
    for h, v in per_diag.items():
        frac = v / N if N else 0
        print(f"  {h} ↔ {mapping.get(h)} : {v}  ({frac:.4%} of labeled)")
