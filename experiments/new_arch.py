"""Candidate architectures vs the baseline MLP, cone dataset, same 70/10/20 split as train_separate.py.
Decisions are made on VALIDATION error; test error is reported alongside."""
import sys, json, time, numpy as np, torch, torch.nn as nn
sys.argv = ['x']; import train_separate as T
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
DS = sys.argv_ds if hasattr(sys, 'argv_ds') else 'data_cone.xlsx'
X, Y = T.load_dataset(DS); tr, va, te = T.split_indices(len(X), 0.2, 0.1, 0)
OUT = 'new_arch_results.json'
try: RES = json.load(open(OUT))
except Exception: RES = {}

def log_feats(Xdf, ref):
    """Standardised log inputs; small offset keeps alpha (min ~1e-5) finite."""
    lo, hi = ref.min().values, ref.max().values
    L = np.log(Xdf.values + 0.01 * (hi - lo))
    Lr = np.log(ref.values + 0.01 * (hi - lo))
    return torch.tensor((L - Lr.mean(0)) / Lr.std(0))

class PowerLawResidual(nn.Module):
    """'Correlation backbone + neural residual':
       t(x) = c + sum_j a_j * log x_j   (a classical power-law correlation, Nu ~ C * prod x_j^a_j in log space)
            + MLP([x_scaled, log x])     (smooth neural correction)
       The linear head is trainable and its exponents a_j are reported."""
    def __init__(s, n_in, n_out, width=128, depth=4):
        super().__init__(); s.n = n_in
        s.lin = nn.Linear(n_in, n_out)
        s.mlp = T.MLP(2 * n_in, n_out, width, depth, 'silu')
    def forward(s, z):                       # z = [x_scaled, logx]
        return s.lin(z[:, s.n:]) + s.mlp(z)

class FixedPlusNet(nn.Module):
    """Prediction = fixed backbone value (precomputed, passed as extra input columns) + scale * MLP."""
    def __init__(s, n_in, n_out, scale, width=128, depth=4):
        super().__init__(); s.n = n_in; s.k = n_out
        s.register_buffer('scale', torch.as_tensor(scale)); s.mlp = T.MLP(n_in, n_out, width, depth, 'silu')
    def forward(s, z):
        return z[:, -s.k:] + s.scale * s.mlp(z[:, :-s.k])

def quad_feats(x):
    cols = [torch.ones(len(x), 1), x]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]): cols.append((x[:, i] * x[:, j]).unsqueeze(1))
    return torch.cat(cols, 1)

def evaluate(name, sc, pred_fn, extra=None, t0=None, hist=None):
    out = {'time_s': time.time() - t0 if t0 else None}
    for k, ix in (('val', va), ('test', te)):
        with torch.no_grad(): P = sc.inv(pred_fn(ix))
        out[k] = {c: T.metrics(Y.values[ix, j], P[:, j]) for j, c in enumerate(Y.columns)}
    if hist: out['final_train_loss'] = hist[-1][2]
    if extra: out.update(extra)
    RES[name] = out; json.dump(RES, open(OUT, 'w'), indent=1)
    v, t = out['val'], out['test']
    print(f"{name:46s} {out['time_s'] or 0:5.0f}s | VAL Nu {v['Nu']['RMSE']:.2e} Sh {v['Sh']['RMSE']:.2e} | "
          f"TEST Nu {t['Nu']['RMSE']:.2e} ({t['Nu']['MaxRelErr_pct']:.2f}%) Sh {t['Sh']['RMSE']:.2e} ({t['Sh']['MaxRelErr_pct']:.2f}%)", flush=True)

def run(which):
    torch.manual_seed(0); t0 = time.time()
    if which == 'baseline':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu']); x, y = sc.x(X), sc.y(Y)
        m = T.MLP(11, 2); h = T.train(m, x[tr], y[tr], x[va], y[va], 2500, 1500, 3e-3, 50)
        evaluate('A0 baseline MLP 4x128 SiLU (log Nu)', sc, lambda ix: m(x[ix]).numpy(), t0=t0, hist=h)
    if which == 'logboth':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        m = T.MLP(11, 2); h = T.train(m, x[tr], y[tr], x[va], y[va], 2500, 1500, 3e-3, 50)
        evaluate('A1 baseline, log Nu AND log Sh', sc, lambda ix: m(x[ix]).numpy(), t0=t0, hist=h)
    if which == 'logfeat':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        z = torch.cat([x, log_feats(X, X.iloc[tr])], 1)
        m = T.MLP(22, 2); h = T.train(m, z[tr], y[tr], z[va], y[va], 2500, 1500, 3e-3, 50)
        evaluate('B  MLP on [x, log x] inputs', sc, lambda ix: m(z[ix]).numpy(), t0=t0, hist=h)
    if which == 'powerlaw':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        z = torch.cat([x, log_feats(X, X.iloc[tr])], 1)
        m = PowerLawResidual(11, 2); h = T.train(m, z[tr], y[tr], z[va], y[va], 2500, 1500, 3e-3, 50)
        # power-law-only fit for reference (how much a classical correlation explains)
        L = z[tr][:, 11:]; A = torch.cat([torch.ones(len(L), 1), L], 1)
        coef = torch.linalg.lstsq(A, y[tr]).solution
        def pl(ix): Lz = z[ix][:, 11:]; return (torch.cat([torch.ones(len(Lz), 1), Lz], 1) @ coef).numpy()
        evaluate('C0 power-law correlation alone (no NN)', sc, pl, t0=t0)
        evaluate('C  power-law backbone + neural residual', sc, lambda ix: m(z[ix]).numpy(), t0=t0, hist=h)
    if which == 'rsm':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        Q = quad_feats(x); coef = torch.linalg.lstsq(Q[tr], y[tr]).solution; base = Q @ coef
        r = y - base; s = r[tr].std(0)
        evaluate('D0 quadratic response surface alone', sc, lambda ix: base[ix].numpy(), t0=t0)
        z = torch.cat([x, base], 1)
        m = FixedPlusNet(11, 2, s);
        h = T.train(m.mlp, x[tr], (r[tr] / s), x[va], (r[va] / s), 2500, 1500, 3e-3, 50)
        evaluate('D  quadratic RSM backbone + scaled NN residual', sc, lambda ix: m(z[ix]).numpy(), t0=t0, hist=h)
    if which == 'multistage':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        m1 = T.MLP(11, 2); T.train(m1, x[tr], y[tr], x[va], y[va], 2500, 1500, 3e-3, 50)
        with torch.no_grad(): r = y - m1(x)
        s = r[tr].std(0)
        torch.manual_seed(1); m2 = T.MLP(11, 2)
        h = T.train(m2, x[tr], r[tr] / s, x[va], r[va] / s, 2500, 1500, 3e-3, 50)
        evaluate('E  multi-stage NN (stage 2 fits scaled residual)', sc, lambda ix: (m1(x[ix]) + s * m2(x[ix])).numpy(),
                 extra={'stage2_scale': s.tolist()}, t0=t0, hist=h)
    if which == 'big':
        sc = T.Scaler().fit(X.iloc[tr], Y.iloc[tr], ['Nu', 'Sh']); x, y = sc.x(X), sc.y(Y)
        m = T.MLP(11, 2, 256, 6); h = T.train(m, x[tr], y[tr], x[va], y[va], 2500, 1500, 3e-3, 50)
        evaluate('F  bigger MLP 6x256 (capacity check)', sc, lambda ix: m(x[ix]).numpy(), t0=t0, hist=h)

if __name__ == '__main__':
    for w in sys.argv_list if hasattr(sys, 'argv_list') else []: run(w)
