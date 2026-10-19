"""PASR-Net: Polynomial-Anchored, Scaled-Residual neural network.

  1. Anchor  : closed-form least-squares polynomial response surface p(x) (degree 2 or 3) on the
               standardised targets (log Nu, Sh). Classical RSM, interpretable, no training.
  2. Encode  : the network sees both the scaled inputs x~ and standardised log inputs (dual encoding).
  3. Residual: the network learns r(x) = t(x) - p(x), divided by the residual's own std s, so its target is O(1).
               Prediction t^ = p(x) + s * NN(x~, log x).
  4. Refine  : optional second residual stage (multi-stage idea of Wang & Lai), same rule.
  5. Ensemble: average of M independently initialised residual networks.
Selection is by VALIDATION error only."""
import sys, json, time, itertools, numpy as np, torch
torch.set_default_dtype(torch.float64)
import train_separate as T


def log_feats(Xdf, ref):
    lo, hi = ref.min().values, ref.max().values
    L = np.log(Xdf.values + 0.01 * (hi - lo)); Lr = np.log(ref.values + 0.01 * (hi - lo))
    return torch.tensor((L - Lr.mean(0)) / Lr.std(0))


def poly_feats(x, degree):
    n = x.shape[1]; cols = [torch.ones(len(x), 1)]
    for d in range(1, degree + 1):
        for c in itertools.combinations_with_replacement(range(n), d):
            v = torch.ones(len(x))
            for i in c: v = v * x[:, i]
            cols.append(v.unsqueeze(1))
    return torch.cat(cols, 1)


class PASR:
    def __init__(self, degree=2, dual=True, stages=1, seeds=1, width=128, depth=4, log_targets=('Nu',),
                 adam=2500, lbfgs=1500, ridge=1e-10):
        self.__dict__.update(locals()); del self.__dict__['self']

    def fit(self, X, Y, tr, va):
        self.sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], list(self.log_targets))
        self.Xref = X.iloc[tr]
        x, y = self.sc.x(X), self.sc.y(Y)
        z = torch.cat([x, log_feats(X, self.Xref)], 1) if self.dual else x
        if self.degree > 0:
            Q = poly_feats(x, self.degree)
            A = Q[tr]; self.coef = torch.linalg.solve(A.T @ A + self.ridge * torch.eye(A.shape[1]), A.T @ y[tr])
            base = Q @ self.coef
        else:
            base = torch.zeros_like(y)
        self.n_poly = int(Q.shape[1]) if self.degree > 0 else 0
        self.stages_ = []; cur = base.clone(); self.hist = []
        for st in range(self.stages):
            r = y - cur; s = r[tr].std(0); nets = []
            for k in range(self.seeds):
                torch.manual_seed(100 * st + k)
                m = T.MLP(z.shape[1], y.shape[1], self.width, self.depth, 'silu')
                self.hist.append(T.train(m, z[tr], r[tr] / s, z[va], r[va] / s, self.adam, self.lbfgs, 3e-3, 50))
                nets.append(m)
            with torch.no_grad(): add = s * torch.stack([m(z) for m in nets]).mean(0)
            self.stages_.append((s, nets)); cur = cur + add
        self._z, self._base = z, base
        return self

    def predict_idx(self, ix):
        with torch.no_grad():
            t = self._base[ix].clone()
            for s, nets in self.stages_: t = t + s * torch.stack([m(self._z[ix]) for m in nets]).mean(0)
        return self.sc.inv(t.numpy())


def report(name, model, X, Y, va, te, t0, store, path):
    out = {'time_s': time.time() - t0, 'n_poly_terms': model.n_poly,
           'stage_scales': [s.tolist() for s, _ in model.stages_]}
    for k, ix in (('val', va), ('test', te)):
        P = model.predict_idx(ix); out[k] = {c: T.metrics(Y.values[ix, j], P[:, j]) for j, c in enumerate(Y.columns)}
    store[name] = out; json.dump(store, open(path, 'w'), indent=1)
    v, t = out['val'], out['test']
    print(f"{name:52s} {out['time_s']:5.0f}s | VAL Nu {v['Nu']['RMSE']:.2e} Sh {v['Sh']['RMSE']:.2e} | "
          f"TEST Nu {t['Nu']['RMSE']:.2e} ({t['Nu']['MaxRelErr_pct']:.2f}%) Sh {t['Sh']['RMSE']:.2e} ({t['Sh']['MaxRelErr_pct']:.2f}%)"
          f" | scales {[np.round(s, 4).tolist() for s in out['stage_scales']]}", flush=True)
    return out


if __name__ == '__main__':
    torch.set_num_threads(2)
    ds = sys.argv[1]; configs = sys.argv[2:]
    X, Y = T.load_dataset(ds); tr, va, te = T.split_indices(len(X), 0.2, 0.1, 0)
    path = f"pasr_{ds.split('.')[0]}.json"
    try: store = json.load(open(path))
    except Exception: store = {}
    C = {
        'q2_x':        dict(degree=2, dual=False),
        'q2_dual':     dict(degree=2, dual=True),
        'q3_dual':     dict(degree=3, dual=True),
        'q2_dual_2st': dict(degree=2, dual=True, stages=2),
        'q0_dual':     dict(degree=0, dual=True),
        'BEST_q?_3seeds': None,
    }
    for c in configs:
        kw = C.get(c) or json.loads(c)
        name = c if c in C else 'custom ' + c
        t0 = time.time(); m = PASR(**kw).fit(X, Y, tr, va)
        report(name, m, X, Y, va, te, t0, store, path)
