import argparse, os, glob
import pandas as pd

def load_contents(contents_root: str):
    """
    把 question_id -> correct_answer 映射出来
    会尝试几条常见路径：
      - <root>/questions.csv
      - <root>/contents/questions.csv
      - <root>/EdNet-Contents/questions.csv
    找不到就返回 {}，后面就只能做曝光分析了
    """
    if not contents_root:
        return {}

    cand_paths = [
        os.path.join(contents_root, "questions.csv"),
        os.path.join(contents_root, "contents", "questions.csv"),
        os.path.join(contents_root, "EdNet-Contents", "questions.csv"),
    ]
    for p in cand_paths:
        if os.path.exists(p):
            q = pd.read_csv(p)
            cols = q.columns.tolist()
            if "question_id" not in cols:
                raise ValueError(f"questions.csv 中没有 question_id, columns={cols}")

            # 猜一下正确答案列名
            ans_col = None
            for name in ["correct_answer", "correct", "answer", "answer_code"]:
                if name in cols:
                    ans_col = name
                    break
            if ans_col is None:
                raise ValueError(f"questions.csv 里找不到正确答案列, columns={cols}")

            mp = dict(zip(q["question_id"].astype(str), q[ans_col].astype(str)))
            return mp

    # 都没找到
    return {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="解压出来的 KT1 目录，比如 data/ednet_full/KT1")
    ap.add_argument("--contents_root", default="", help="题库所在目录，比如 data/ednet_contents/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 1) 先尝试加载题库
    qid2ans = load_contents(args.contents_root)
    has_contents = len(qid2ans) > 0
    if has_contents:
        print(f"✅ loaded contents ({len(qid2ans)} questions) from {args.contents_root}")
    else:
        print("⚠️ 没有找到 contents/questions.csv → 会写出 correct=NA")

    # 2) 扫用户文件
    pattern = os.path.join(args.root, "u*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[*] found {len(files)} user files under {args.root}")

    rows = []
    for i, fpath in enumerate(files, 1):
        df = pd.read_csv(fpath)
        cols = df.columns.tolist()

        # 识别题目列
        if "question_id" in cols:
            item_col = "question_id"
        elif "problem_id" in cols:
            item_col = "problem_id"
        else:
            raise ValueError(f"{fpath} 里找不到 question_id / problem_id, columns={cols}")

        # 识别学生答案列
        if "user_answer" in cols:
            ans_col = "user_answer"
        elif "user_response" in cols:
            ans_col = "user_response"
        else:
            raise ValueError(f"{fpath} 里找不到 user_answer / user_response, columns={cols}")

        # 时间戳列
        if "timestamp" in cols:
            ts_col = "timestamp"
        elif "start_time" in cols:
            ts_col = "start_time"
        else:
            ts_col = None

        uid = os.path.splitext(os.path.basename(fpath))[0]  # u12345 → user_id

        for r in df.itertuples(index=False):
            qid = str(getattr(r, item_col))
            stu_ans = str(getattr(r, ans_col))
            ts = getattr(r, ts_col) if ts_col else None

            if has_contents and qid in qid2ans:
                gold = qid2ans[qid]
                correct = 1 if stu_ans == gold else 0
            else:
                gold = None
                correct = None

            rows.append({
                "user_id": uid,
                "item_id": qid,
                "student_answer": stu_ans,
                "gold_answer": gold,
                "correct": correct,
                "timestamp": ts,
            })

        if i % 200 == 0:
            print(f"  ... processed {i}/{len(files)} users")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)}")
    if not has_contents:
        print("⚠️ 没有 contents → 先这样用，等你把题库下好再跑一遍。")
