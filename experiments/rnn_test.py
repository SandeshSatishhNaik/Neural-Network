"""Does an RNN help on this data? Same data, split, scaling and training budget as train_separate.py."""
import sys, time, json, numpy as np, torch, torch.nn as nn
sys.argv=['x']; import train_separate as T
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
X,Y=T.load_dataset('data_cone.xlsx'); tr,va,te=T.split_indices(len(X),0.2,0.1,0)
sc=T.Scaler().fit(X.iloc[tr],Y.iloc[tr],['Nu']); x,y=sc.x(X),sc.y(Y)

class SeqRNN(nn.Module):
    """Paper-style: tanh recurrent layer, linear output. The 11 parameters are fed ONE PER STEP
    (the only way to give an RNN a 'sequence' when each row is a single parameter combination)."""
    def __init__(s, kind='rnn', h=128, order=None):
        super().__init__(); s.order=order
        cell={'rnn':lambda:nn.RNN(1,h,batch_first=True,nonlinearity='tanh'),'gru':lambda:nn.GRU(1,h,batch_first=True),'lstm':lambda:nn.LSTM(1,h,batch_first=True)}[kind]
        s.r=cell(); s.out=nn.Linear(h,2)
    def forward(s,x):
        if s.order is not None: x=x[:,s.order]
        o,_=s.r(x.unsqueeze(-1)); return s.out(o[:,-1])
class Len1RNN(nn.Module):
    """All 11 parameters given at once, sequence length 1 (h0 = 0): mathematically a 1-hidden-layer tanh MLP."""
    def __init__(s,h=128): super().__init__(); s.r=nn.RNN(11,h,batch_first=True,nonlinearity='tanh'); s.out=nn.Linear(h,2)
    def forward(s,x): o,_=s.r(x.unsqueeze(1)); return s.out(o[:,-1])

rng=np.random.default_rng(1); perm=torch.tensor(rng.permutation(11))
models={
 'MLP 4x128 SiLU (your model)':lambda:T.MLP(11,2,128,4,'silu'),
 'RNN tanh, params as sequence (file order)':lambda:SeqRNN('rnn'),
 'RNN tanh, params as sequence (shuffled order)':lambda:SeqRNN('rnn',order=perm),
 'GRU, params as sequence':lambda:SeqRNN('gru'),
 'RNN tanh, sequence length 1 (= 1-layer tanh MLP)':lambda:Len1RNN(),
}
res={}
for name,mk in models.items():
    torch.manual_seed(0); m=mk(); t0=time.time()
    h=T.train(m,x[tr],y[tr],x[va],y[va],2500,1500,3e-3,log_every=50)
    with torch.no_grad(): p=sc.inv(m(x[te]).numpy())
    yt=Y.values[te]; r={c:T.metrics(yt[:,j],p[:,j]) for j,c in enumerate(Y.columns)}
    r['params']=sum(q.numel() for q in m.parameters()); r['sec']=time.time()-t0; r['train_loss']=h[-1][2]; r['val_loss']=h[-1][3]
    res[name]=r; json.dump(res,open('rnn_results.json','w'),indent=1)
    print(f"{name:52s} params {r['params']:6d} {r['sec']:5.0f}s | Nu R2 {r['Nu']['R2']:.6f} RMSE {r['Nu']['RMSE']:.2e} maxRel {r['Nu']['MaxRelErr_pct']:.2f}% | Sh R2 {r['Sh']['R2']:.6f} RMSE {r['Sh']['RMSE']:.2e} maxRel {r['Sh']['MaxRelErr_pct']:.2f}% | loss tr {r['train_loss']:.1e} va {r['val_loss']:.1e}",flush=True)
