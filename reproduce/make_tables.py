"""Tables 2-4 from a user-provided per-run results CSV: task means over the seeds, macro-averages,
and paired two-sided exact Wilcoxon signed-rank tests on the task-level differences with Benjamini-Hochberg correction within each comparison.
  python make_tables.py <per-run-results.csv>"""
import argparse, collections, csv
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results_csv', help='Per-run results CSV')
PATH = parser.parse_args().results_csv
M5 = ('acc', 'bacc', 'mcc', 'roc_auc', 'pr_auc')
TABLES = {'Table 2': ('BP', 'PDeepPP', ('UniDL4BioPep',)), 'Table 3': ('PTM', 'PDeepPP', ('MusiteDeep', 'PhosF3C'))}
TABLE4 = (('BP', 'ESM-2 linear probe'), ('BP', 'ESM-2 MLP'), ('BP', 'UniDL4BioPep'), ('PTM', 'MusiteDeep'), ('PTM', 'PhosF3C'))


def wilcoxon_exact(d):
    d = d[d != 0]; n = len(d)
    if n == 0: return 1.0
    a = np.abs(d); order = a.argsort(); ranks = np.empty(n); i = 0; sa = a[order]
    while i < n:
        j = i
        while j + 1 < n and abs(sa[j + 1] - sa[i]) < 1e-12: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    r2 = np.rint(ranks * 2).astype(int); tot = r2.sum(); w = int(r2[d > 0].sum())
    counts = np.zeros(tot + 1); counts[0] = 1
    for x in r2:
        new = counts.copy(); new[x:] += counts[:tot + 1 - x]; counts = new
    cdf = np.cumsum(counts) / counts.sum()
    return float(min(1.0, 2 * min(cdf[w], 1 - (cdf[w - 1] if w > 0 else 0))))


def bh(ps):
    ps = np.array(ps); o = ps.argsort(); q = np.empty(len(ps)); prev = 1.0
    for rank, i in zip(range(len(ps), 0, -1), o[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
    return q


runs = collections.defaultdict(list)
for r in csv.DictReader(open(PATH)):
    runs[(r['family'], r['model'], r['task'])].append(r)
means = {k: {m: float(np.mean([float(x[m]) for x in v])) for m in M5} for k, v in runs.items()}


def task_means(fam, model):
    return {t: v for (f, m, t), v in means.items() if f == fam and m == model}


for name, (fam, ref, bases) in TABLES.items():
    print(f'\n{name} (test-set means over the seeds): task, then PR AUC / MCC / BACC of ' + ', '.join((ref,) + bases))
    a = task_means(fam, ref)
    for t in sorted(a, key=str.lower):
        print(f'{t:32s}' + '  '.join('/'.join(f'{task_means(fam, x)[t][k]:.3f}' for k in ('pr_auc', 'mcc', 'bacc')) for x in (ref,) + bases))
    print(f'{"Macro-average":32s}' + '  '.join('/'.join(f'{np.mean([v[k] for v in task_means(fam, x).values()]):.3f}' for k in ('pr_auc', 'mcc', 'bacc')) for x in (ref,) + bases))

print('\nTable 4 (paired tests of PDeepPP minus baseline): comparison, metric, median difference, p, q')
for fam, base in TABLE4:
    a, b = task_means(fam, 'PDeepPP'), task_means(fam, base)
    d = {m: np.array([a[t][m] - b[t][m] for t in sorted(a)]) for m in M5}
    p = [wilcoxon_exact(d[m]) for m in M5]
    for m, pv, qv in zip(M5, p, bh(p)):
        print(f'{base:20s} {fam:4s} {m:8s} {np.median(d[m]):+.3f}  {pv:.3g}  {qv:.3g}')
