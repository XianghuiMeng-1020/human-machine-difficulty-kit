import argparse, csv, os

def load_cross(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            h = row.get("\ufeffH_label") or row.get("H_label") or row.get("H") or ""
            m_easy = int(row.get("简单M", 0))
            m_mid  = int(row.get("中等M", 0))
            m_hard = int(row.get("困难M", 0))
            rows.append((h, m_easy, m_mid, m_hard))
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cross", required=True)
    args=ap.parse_args()

    rows = load_cross(args.cross)

    total = 0
    human_counts = {"简单H":0,"中等H":0,"困难H":0,"MissingH":0}
    model_counts = {"简单M":0,"中等M":0,"困难M":0}
    aligned = 0
    labeled_total = 0

    for h, me, mm, mh in rows:
        s = me+mm+mh
        total += s
        if h in human_counts:
            human_counts[h] += s
        else:
            human_counts["MissingH"] += s

        model_counts["简单M"] += me
        model_counts["中等M"] += mm
        model_counts["困难M"] += mh

        if h in ("简单H","中等H","困难H"):
            labeled_total += s
            if h == "简单H":
                aligned += me
            elif h == "中等H":
                aligned += mm
            elif h == "困难H":
                aligned += mh

    print("File:", args.cross)
    print("total cells:", total)
    print("human counts:", human_counts)
    print("model counts:", model_counts)
    if total>0:
        print("overall diagonal ratio:", aligned/total)
    if labeled_total>0:
        print("labeled-only diagonal ratio:", aligned/labeled_total)
        print("labeled_total:", labeled_total)
        print("missing portion:", (total - labeled_total)/total)
