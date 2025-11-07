
import os, pandas as pd, numpy as np
from pathlib import Path

Path('tables').mkdir(exist_ok=True)

def _read_csv(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def _to_num(s):
    return pd.to_numeric(s, errors='coerce')

def _misalignment_summary(df, dataset_name):
    # Expect columns: qid, model, is_correct, (optional) diff_conf, (optional) human_label, (optional) conf
    if df.empty:
        out = pd.DataFrame(columns=['dataset','model','#human_easy__model_hard','#human_hard__model_easy','n_items'])
        out.to_csv(f"tables/{dataset_name}_misalignment_summary.csv", index=False)
        return

    df = df.copy()
    df['qid'] = df['qid'].astype(str)
    df['model'] = df['model'].astype(str)
    df['is_correct'] = _to_num(df.get('is_correct'))
    df['conf'] = _to_num(df.get('conf'))
    # Use per-(qid,model) aggregation
    g = (df.groupby(['qid','model'], as_index=False)
            .agg(is_correct=('is_correct','mean'),
                 conf=('conf','mean'),
                 diff_conf=('diff_conf', lambda s: _to_num(s).mean() if 'diff_conf' in df.columns else np.nan),
                 human_label=('human_label', lambda s: s.iloc[0] if 'human_label' in df.columns else np.nan)
                 ))
    g['model_err'] = 1 - g['is_correct']

    # Human difficulty flags (graceful fallback)
    human_easy = pd.Series([False]*len(g))
    human_hard = pd.Series([False]*len(g))

    if 'diff_conf' in g.columns and g['diff_conf'].notna().any():
        # numeric confidence that item is HARD for humans (0=easy,1=hard)
        dc = _to_num(g['diff_conf'])
        human_easy = dc <= 0.25
        human_hard = dc >= 0.75
    elif 'human_label' in g.columns and g['human_label'].notna().any():
        hl = g['human_label'].astype(str).str.lower()
        human_easy = hl.eq('low') | hl.eq('easy')
        human_hard = hl.eq('high') | hl.eq('hard')

    # Summaries by model
    rows = []
    for m, sub in g.groupby('model', dropna=True):
        he_mh = int(((human_easy) & (sub['model_err'] >= 0.5)).sum())
        hh_me = int(((human_hard) & (sub['model_err'] < 0.5)).sum())
        rows.append({
            'dataset': dataset_name,
            'model': m,
            '#human_easy__model_hard': he_mh,
            '#human_hard__model_easy': hh_me,
            'n_items': int(sub['qid'].nunique())
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"tables/{dataset_name}_misalignment_summary.csv", index=False)

def _risk_coverage(df, dataset_name):
    # Risk–coverage: sort by conf desc; track cumulative acc
    if df.empty:
        pd.DataFrame(columns=['dataset','model','covered','cum_acc','mean_conf']).to_csv(
            f"tables/{dataset_name}_risk_coverage.csv", index=False)
        return
    df = df.copy()
    df['model'] = df['model'].astype(str)
    df['is_correct'] = _to_num(df.get('is_correct'))
    df['conf'] = _to_num(df.get('conf'))

    rows=[]
    for m, sub in df.groupby('model', dropna=True):
        kk = sub.dropna(subset=['conf','is_correct']).copy()
        kk = kk.sort_values('conf', ascending=False)
        if kk.empty:
            rows.append({'dataset':dataset_name,'model':m,'covered':0,'cum_acc':'','mean_conf':''})
            continue
        kk['rank'] = np.arange(1, len(kk)+1)
        kk['covered'] = kk['rank'] / len(kk)
        kk['cum_acc'] = kk['is_correct'].expanding().mean()
        kk['mean_conf'] = kk['conf'].expanding().mean()
        kk['dataset'] = dataset_name
        rows += kk[['dataset','model','covered','cum_acc','mean_conf']].to_dict('records')
    pd.DataFrame(rows).to_csv(f"tables/{dataset_name}_risk_coverage.csv", index=False)

def load_and_group(path, dataset_name):
    df = _read_csv(path)
    if df.empty:
        # write empty outputs to keep pipeline consistent
        _misalignment_summary(df, dataset_name)
        _risk_coverage(df, dataset_name)
        return
    # Minimal columns normalization
    need = {'qid','model','is_correct'}
    missing = need - set(df.columns)
    if missing:
        # try to infer model if absent
        if 'model' not in df.columns:
            df['model'] = 'unknown'
        if 'qid' not in df.columns or 'is_correct' not in df.columns:
            # cannot continue; write empties
            _misalignment_summary(pd.DataFrame(), dataset_name)
            _risk_coverage(pd.DataFrame(), dataset_name)
            return
    _misalignment_summary(df, dataset_name)
    _risk_coverage(df, dataset_name)

if __name__ == '__main__':
    Path('tables').mkdir(exist_ok=True)
    # RACE
    load_and_group('data/race_runs.csv', 'race')
    # EEDI (tolerate missing diff_conf)
    load_and_group('data/eedi_runs.csv', 'eedi')
