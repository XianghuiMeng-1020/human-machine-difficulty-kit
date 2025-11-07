import argparse, csv

def load_matrix(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    cols = rows[0][1:]
    data = {rows[i][0]: {cols[j]: int(rows[i][1:][j]) for j in range(len(cols))}
            for i in range(1, len(rows))}
    return data, cols

def compute_alignment(table, cols, mapping):
    # mapping: dict of {human_label -> model_label} viewed as "aligned"
    N = sum(sum(row.values()) for row in table.values())
    diag = 0
    per_diag = {}
    for h, row in table.items():
        m_aligned = mapping.get(h)
        if m_aligned in row:
            diag += row[m_aligned]
            per_diag[h] = row[m_aligned]
        else:
            per_diag[h] = 0
    return N, diag, per_diag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", required=True, help="human_x_model csv")
    ap.add_argument("--map", default="简单H=简单M,中等H=中等M,困难H=困难M",
                    help="alignment mapping, comma separated pairs H=M")
    args = ap.parse_args()

    table, cols = load_matrix(args.cross)
    mapping = {}
    for pair in args.map.split(","):
        h, m = pair.split("=")
        mapping[h] = m

    N, diag, per_diag = compute_alignment(table, cols, mapping)

    print(f"File: {args.cross}")
    print(f"N = {N}")
    print(f"Aligned (diagonal) = {diag}")
    print(f"Alignment Index = {diag / N if N else 0:.4f}")
    for h, v in per_diag.items():
        frac = v / N if N else 0
        print(f"  {h} ↔ {mapping.get(h)} : {v}  ({frac:.4%} of all)")
