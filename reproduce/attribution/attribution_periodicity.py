"""Three-residue periodicity of the PTM attribution profiles (Reviewer 4, Major comment 8): positions grouped by position mod 3, range of the three group means of the
seed-mean signed IG of positive windows (random-negative baseline), against 20,000 random permutations of the 32 non-central positions.
Reads fig_data.json and writes periodicity.csv in the directory given by ATTR_OUT."""
import csv, json, os
import numpy as np
E = os.environ.get('ATTR_OUT', '.')  # directory with fig_data.json (written by interpret_collect.py)
f = json.load(open(f'{E}/fig_data.json')); POS = np.arange(-16, 17); nc = POS != 0; p = POS[nc]; rng = np.random.default_rng(0); N = 20000
with open(f'{E}/periodicity.csv', 'w', newline='') as h:
    w = csv.writer(h); w.writerow(['task', 'phase0', 'phase1', 'phase2', 'best_phase', 'perm_count_ge', 'n_perm'])
    for t in f['ptm_tasks']:
        v = np.array(f['ptm_profile_e8'][t], float)[nc]; ph = np.array([v[p % 3 == k].mean() for k in range(3)]) / np.abs(v).max(); stat = np.ptp(ph)
        c = sum(np.ptp([x[p % 3 == k].mean() for k in range(3)]) / np.abs(v).max() >= stat for x in (rng.permutation(v) for _ in range(N)))
        w.writerow([t, *np.round(ph, 3), int(ph.argmax()), int(c), N]); print(t, np.round(ph, 2), int(ph.argmax()), c)
