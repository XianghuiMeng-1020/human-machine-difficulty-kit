import argparse, os, random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
这是一个“CL4KT 影子版”：
- sequence input from 00_make_sessions_from_ednet.py
- item difficulty from 01_build_item_difficulty_table.py
- model = item_emb + diff_emb → MLP → p(correct)
我们只训练一个 epoch，拿到一个 AUC / ACC 就行。

重点不是分数，而是：有 diff_emb vs 无 diff_emb 的对比 → 证明 KT 这一线也无法把人机对齐吃干净。
"""


class KTSeqDataset(Dataset):
    def __init__(self, sessions_csv, item_diff_csv, max_len=100):
        self.sessions = pd.read_csv(sessions_csv)
        self.item_diff = pd.read_csv(item_diff_csv).set_index("item_id")
        self.max_len = max_len

        # 重新映射 item_id → 连续 idx
        all_item_ids = set()
        for row in self.sessions.itertuples(index=False):
            for it in str(row.items).split():
                all_item_ids.add(it)
        self.item2idx = {it: i+1 for i, it in enumerate(sorted(all_item_ids))}
        self.n_items = len(self.item2idx) + 1  # +1 for padding 0

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        row = self.sessions.iloc[idx]
        items = str(row.items).split()
        corrs = str(row.corrects).split()
        # 截断 / pad 到 max_len
        items = items[-self.max_len:]
        corrs = corrs[-self.max_len:]
        L = len(items)
        x_items = np.zeros(self.max_len, dtype=np.int64)
        x_corrs = np.zeros(self.max_len, dtype=np.float32)
        x_diffs = np.zeros(self.max_len, dtype=np.int64)  # -1 表示没有diff

        for i, (it, c) in enumerate(zip(items, corrs)):
            x_items[self.max_len - L + i] = self.item2idx[it]
            x_corrs[self.max_len - L + i] = float(c)
            # difficulty
            if it in self.item_diff.index:
                di = int(self.item_diff.loc[it, "diff_int"])
            else:
                di = -1
            x_diffs[self.max_len - L + i] = di

        # 预测目标是最后一个位置的 correctness
        tgt = x_corrs[-1]
        return (
            torch.tensor(x_items),
            torch.tensor(x_corrs),
            torch.tensor(x_diffs),
            torch.tensor(tgt, dtype=torch.float32),
        )


class CL4KTStub(nn.Module):
    def __init__(self, n_items, diff_vocab=4, d_model=64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d_model)
        self.diff_emb = nn.Embedding(diff_vocab, d_model)  # 0,1,2,3(=no diff)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, item_ids, diff_ids):
        # item_ids: (B, T)
        # diff_ids: (B, T)
        it_e = self.item_emb(item_ids)          # (B,T,D)
        df_e = self.diff_emb(diff_ids)          # (B,T,D)
        # 只用最后一个时间步
        it_last = it_e[:, -1, :]
        df_last = df_e[:, -1, :]
        x = torch.cat([it_last, df_last], dim=-1)
        p = self.fc(x).squeeze(-1)
        return p


def train_one(sessions_csv, item_diff_csv, out_dir, max_len=100, batch_size=64, use_diff=True):
    ds = KTSeqDataset(sessions_csv, item_diff_csv, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = CL4KTStub(ds.n_items, diff_vocab=4, d_model=64)
    if not use_diff:
        # 把 diff_emb 固定成 0，等价于不用
        with torch.no_grad():
            model.diff_emb.weight.zero_()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        total_n = 0
        for it, corr, diff, tgt in dl:
            # 把 -1 的 diff 先转成 3 这个 index
            diff = diff.clone()
            diff[diff < 0] = 3
            p = model(it, diff)
            loss = bce(p, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(tgt)
            total_n += len(tgt)
        print(f"[use_diff={use_diff}] epoch {epoch+1} loss={total_loss/total_n:.4f}")

    # 简单评估
    model.eval()
    with torch.no_grad():
        all_p, all_y = [], []
        for it, corr, diff, tgt in dl:
            diff = diff.clone()
            diff[diff < 0] = 3
            p = model(it, diff)
            all_p.extend(p.tolist())
            all_y.extend(tgt.tolist())
    all_p = np.array(all_p)
    all_y = np.array(all_y)
    acc = ((all_p >= 0.5) == (all_y >= 0.5)).mean()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"cl4kt_stub_useDiff{int(use_diff)}.pt"))
    print(f"✅ saved model to {out_dir}, acc={acc:.4f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="baselines/cl4kt_diff/ednet_u200_sessions.csv")
    ap.add_argument("--items", default="baselines/cl4kt_diff/ednet_u200_items.csv")
    ap.add_argument("--out_dir", default="baselines/cl4kt_diff/out")
    args = ap.parse_args()

    acc_with = train_one(args.sessions, args.items, args.out_dir, use_diff=True)
    acc_nodf = train_one(args.sessions, args.items, args.out_dir, use_diff=False)

    # 写个对比表
    out_csv = os.path.join(args.out_dir, "cl4kt_stub_vs_mvhmda.csv")
    pd.DataFrame([
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-stub + diff", "acc": acc_with},
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-stub w/o diff", "acc": acc_nodf},
        {"dataset": "EdNet-KT1(u200)", "method": "MV-HMDA-proxy (yours)", "acc": df_acc_placeholder if False else None},
    ]).to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv}")
