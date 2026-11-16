"""Audit the dataset released in GitHub commit 1c3daa4 (= arXiv v6 data).

Usage: python audit_legacy_github_data.py <repo>/data <out.csv>
Reports per task: train/test positive and negative counts, fraction of test
sequences that appear verbatim in the training file, and raw-length profile
(PTM: share of windows with exactly 33 characters, by class).
"""
import collections, csv, os, sys
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
rows = []
for task in sorted(os.listdir(root)):
    p = os.path.join(root, task)
    if not os.path.isdir(p):
        continue
    fs = [f for f in os.listdir(p) if f.endswith('.xlsx') and not f.startswith('~$')]
    tr = [f for f in fs if 'train' in f.lower()][0]
    te = [f for f in fs if 'test' in f.lower()][0]
    a, b = pd.read_excel(os.path.join(p, tr)), pd.read_excel(os.path.join(p, te))
    sa = a.sequence.astype(str).str.strip().str.upper()
    sb = b.sequence.astype(str).str.strip().str.upper()
    ca, cb = collections.Counter(a.label), collections.Counter(b.label)
    rlen_pos, rlen_neg = sb[b.label == 1].str.len(), sb[b.label == 0].str.len()
    rows.append(dict(
        task=task, train_file=tr, test_file=te,
        train_pos=ca[1], train_neg=ca[0], test_pos=cb[1], test_neg=cb[0],
        test_in_train_frac=round(sb.isin(set(sa)).mean(), 4),
        test_pos_in_train_frac=round(sb[b.label == 1].isin(set(sa)).mean(), 4),
        test_pos_len_median=int(rlen_pos.median()), test_pos_len_max=int(rlen_pos.max()),
        test_neg_len_median=int(rlen_neg.median()), test_neg_len_max=int(rlen_neg.max()),
        test_pos_len33_frac=round((rlen_pos == 33).mean(), 4),
        test_neg_len33_frac=round((rlen_neg == 33).mean(), 4),
    ))
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f'{len(rows)} tasks -> {out}')
