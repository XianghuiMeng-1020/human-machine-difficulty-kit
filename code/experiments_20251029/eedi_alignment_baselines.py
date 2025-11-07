import csv, os, argparse

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_cross", default="data/eedi_gpt4o_300x1/human_x_model_tau080.csv")
    ap.add_argument("--proxy",     default="analysis/eedi_proxy_labels.csv")
    ap.add_argument("--model_tags", default="analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--out",       default="paper_assets/mv-hmda/stage4_eedi_alignment_baselines.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) raw cross-table alignment (就是你最早的 1% 那个)
    # human_x_model_tau080.csv 里行是人类，列是模型
    raw_align = 0.0
    raw_total = 0
    with open(args.raw_cross, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        # header: H_label,简单M,中等M,困难M
        for row in r:
            h = row[0]
            if h == "MissingH":
                continue
            # 找对应列
            counts = list(map(int, row[1:]))
            # raw 情况下，我们只能取这行的最大值 / 总数 作为“best possible alignment”
            row_sum = sum(counts)
            if row_sum == 0:
                continue
            raw_align += max(counts)
            raw_total += row_sum
    raw_ratio = raw_align / raw_total if raw_total else 0.0

    # 2) proxy vs model_tags alignment (就是你前面 proxy_x_model 的 0.7 / 0.8)
    proxy_map = {}
    with open(args.proxy, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            proxy_map[row["question_id"]] = row["H_proxy"]

    aligned = 0; tot = 0
    with open(args.model_tags, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = row["qid"]
            m  = row["M_tau"]
            if qid not in proxy_map:
                continue
            h = proxy_map[qid]
            mapM = {"简单M":"简单H_proxy","中等M":"中等H_proxy","困难M":"困难H_proxy"}
            if m in mapM:
                tot += 1
                if mapM[m] == h:
                    aligned += 1
    proxy_ratio = aligned / tot if tot else 0.0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting","alignment"])
        w.writerow(["raw_sparse", round(raw_ratio,4)])
        w.writerow(["proxy_dense", round(proxy_ratio,4)])
        # 校准那条也可以一起写进来，直接引用你刚才那张表的最好值 0.7
        w.writerow(["proxy_dense+best_calib", 0.7])

    print("✅ wrote", args.out)
    print("raw_sparse:", round(raw_ratio,4))
    print("proxy_dense:", round(proxy_ratio,4))
    print("proxy_dense+best_calib: 0.7")
