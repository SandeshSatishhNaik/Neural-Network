"""Confirmation run: baseline vs PASR variants, one model per dataset per target, 3 seeds each.
Reports per-seed val/test RMSE (mean +- sd) and the 3-seed ensemble."""
import sys, json, time, numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
import train_separate as T, pasr as P

CONF = {'Nu': {'baseline': dict(degree=0, dual=False), 'PASR quad anchor': dict(degree=2, dual=False)},
        'Sh': {'baseline': dict(degree=0, dual=False), 'PASR dual encoding': dict(degree=0, dual=True)}}
path = 'confirm.json'
try: R = json.load(open(path))
except Exception: R = {}
for ds in sys.argv[1:]:
    X, Y = T.load_dataset(ds); tr, va, te = T.split_indices(len(X), 0.2, 0.1, 0); name = ds.split('.')[0]
    for tgt, confs in CONF.items():
        Yt = Y[[tgt]]
        for cname, kw in confs.items():
            key = f'{name}|{tgt}|{cname}'
            if key in R: continue
            preds = {'val': [], 'test': []}; per = []
            for s in range(3):
                t0 = time.time(); torch.manual_seed(s)
                m = P.PASR(log_targets=('Nu',), **kw)
                # vary the seed of the residual net
                orig = torch.manual_seed
                m.fit.__func__  # noqa
                import types
                def seeded_fit(self, X, Y, tr, va, s=s):
                    _ms = torch.manual_seed
                    torch.manual_seed = lambda v, _ms=_ms: _ms(v + 1000 * s)
                    try: return P.PASR.fit(self, X, Y, tr, va)
                    finally: torch.manual_seed = _ms
                m.fit = types.MethodType(seeded_fit, m); m.fit(X, Yt, tr, va)
                row = {}
                for k, ix in (('val', va), ('test', te)):
                    p = m.predict_idx(ix)[:, 0]; preds[k].append(p)
                    row[k] = T.metrics(Yt.values[ix, 0], p)
                per.append(row)
                print(f'{key:40s} seed {s} {time.time()-t0:4.0f}s  val {row["val"]["RMSE"]:.3e}  test {row["test"]["RMSE"]:.3e} ({row["test"]["MaxRelErr_pct"]:.2f}%)', flush=True)
            ens = {k: T.metrics(Yt.values[ix, 0], np.mean(preds[k], 0)) for k, ix in (('val', va), ('test', te))}
            R[key] = {'per_seed': per, 'ensemble': ens,
                      'val_rmse_mean': float(np.mean([r['val']['RMSE'] for r in per])), 'val_rmse_sd': float(np.std([r['val']['RMSE'] for r in per])),
                      'test_rmse_mean': float(np.mean([r['test']['RMSE'] for r in per])), 'test_rmse_sd': float(np.std([r['test']['RMSE'] for r in per]))}
            json.dump(R, open(path, 'w'), indent=1)
            print(f'== {key}: val {R[key]["val_rmse_mean"]:.3e}+-{R[key]["val_rmse_sd"]:.1e}  test {R[key]["test_rmse_mean"]:.3e}+-{R[key]["test_rmse_sd"]:.1e}  | ensemble test {ens["test"]["RMSE"]:.3e} ({ens["test"]["MaxRelErr_pct"]:.2f}%)', flush=True)
