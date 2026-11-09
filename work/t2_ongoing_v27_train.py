#!/usr/bin/env python
"""Train-only IoU fine tuning of the seed-1 ongoing RNN and batched inference.

The architecture and [T-60,T) information contract match t2_rnn.py.  Fine tuning
adds Lovasz's per-window Jaccard surrogate, a small learning rate, and averages
the final three epochs.  No validation/private labels enter optimization.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
import t2_cnn as C
import t2_rnn as R
import t2_pseudo as P
import t2_ongoing_v27 as E
from evaluate import ROOT, T2_PANELS

MODEL = ROOT / "work" / "models" / "t2_ongoing_v27_lovasz.pt"


def model_sha256():
    digest = hashlib.sha256()
    with MODEL.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_prob(panel, split):
    path = E.CACHE/f"lovasz_{panel}_{split}.npy"
    source = path.with_suffix(".json")
    expected = model_sha256()
    if not source.exists() or json.loads(source.read_text()).get("model_sha256")!=expected:
        raise ValueError(f"Stale or unversioned predictions: {path}; run cache for the current model")
    return np.load(path)


def checked_selection():
    selected = json.loads((E.OUT/"lovasz_selected.json").read_text())
    if selected.get("model_sha256")!=model_sha256():
        raise ValueError("Selection belongs to different model weights; rebuild cache and select again")
    return selected


def lovasz_probability(prob, target, mask):
    """Per-window Lovasz extension of intersection-over-union loss."""
    losses = []
    for pr, y, m in zip(prob.flatten(1), target.flatten(1), mask.flatten(1)):
        pr, y = pr[m.bool()], y[m.bool()]
        if len(y) == 0:
            continue
        errors, order = (pr-y).abs().sort(descending=True)
        ys = y[order]
        total = ys.sum()
        intersection = total-ys.cumsum(0)
        union = total+(1-ys).cumsum(0)
        jaccard = 1-intersection/union.clamp(min=1)
        grad = torch.cat([jaccard[:1], jaccard[1:]-jaccard[:-1]])
        losses.append(torch.dot(errors, grad))
    return torch.stack(losses).mean()


def cmd_train(args):
    torch.manual_seed(2033)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(2033)
    data = {p: C.PanelData(p) for p in T2_PANELS}
    wins = {p: d.windows() for p,d in data.items()}
    model = R.build_model().cuda()
    model.load_state_dict(torch.load(R.MODELS / "t2_rnn_s1.pt", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.samples//args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*steps, eta_min=args.lr*.05)
    states = []
    print(json.dumps(vars(args)), flush=True)
    for ep in range(args.epochs):
        t0, totals = time.time(), np.zeros(3)
        model.train()
        for it in range(steps):
            panel = T2_PANELS[(it+ep)%len(T2_PANELS)]
            D, W = data[panel], wins[panel]
            xs, ys, ms = [], [], []
            for j in rng.integers(0,len(W),args.batch):
                d,T = W[j]
                h = slice(T-12,T)
                hide = rng.random((12,D.n)) < rng.uniform(0,.5)
                xs.append(C.make_x(D.V[d,h],D.Q[d,h],D.O[d,h],D.static_ch,D.st["ffs"],D.st["cap"],
                                   D.vcut,T,int(D.weekday[d]),hide))
                fut = D.V[d,T+1:T+7].astype(np.float32)
                m = np.isfinite(fut)
                with np.errstate(invalid="ignore"):
                    ys.append((m & (fut<=D.vcut)).astype(np.float32))
                ms.append(m.astype(np.float32))
            X,Y,M = (v.cuda() for v in R.to_batch(xs,ys,ms))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logit = model(X)
            logit = logit.float()
            prob = logit.sigmoid()
            bce = (F.binary_cross_entropy_with_logits(logit,Y,reduction="none")*M).sum()/M.sum().clamp(min=1)
            pr = prob*M
            soft = 1-(((pr*Y).sum((1,2))+1)/((pr+Y-pr*Y).sum((1,2))+1)).mean()
            lovasz = lovasz_probability(prob,Y,M)
            loss = bce+.5*soft+.5*lovasz
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            sched.step()
            totals += [bce.item(),soft.item(),lovasz.item()]
        if ep>=args.epochs-3:
            states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        print(f"epoch {ep+1}/{args.epochs} bce/soft/lovasz {totals/steps} {time.time()-t0:.1f}s",flush=True)
    averaged = {k:torch.stack([s[k] for s in states]).mean(0) for k in states[0]}
    torch.save(averaged,MODEL)
    MODEL.with_suffix(".json").write_text(json.dumps(dict(**vars(args),seed=2033,source="t2_rnn_s1.pt",
           source_split="train",information="[T-60,T)",average_last_epochs=len(states)),indent=2))
    print("saved",MODEL,flush=True)


def load_model(expected_sha=None):
    if expected_sha is not None and expected_sha!=model_sha256():
        raise ValueError("Requested model SHA256 does not match current checkpoint")
    model = R.build_model().cuda()
    model.load_state_dict(torch.load(MODEL,weights_only=True,map_location="cuda"))
    return model.eval()


def cmd_cache(args):
    fingerprint = model_sha256()
    model = load_model(fingerprint)
    for split in ("validation","private"):
        for panel in T2_PANELS:
            start = time.time()
            path = E.CACHE/f"lovasz_{panel}_{split}.npy"
            source = path.with_suffix(".json")
            if path.exists() and source.exists() and json.loads(source.read_text()).get("model_sha256")==fingerprint:
                continue
            ws = [w for w in P.windows(panel,split) if w.cond=="ongoing"]
            D = C.PanelData(panel,split,unmasked=False)
            out = np.zeros((len(ws),6,D.n),np.float32)
            for start_i in range(0,len(ws),32):
                xs = [C.make_x(D.V[w.day,w.T-12:w.T],D.Q[w.day,w.T-12:w.T],D.O[w.day,w.T-12:w.T],
                       D.static_ch,D.st["ffs"],D.st["cap"],D.vcut,w.T,w.weekday) for w in ws[start_i:start_i+32]]
                out[start_i:start_i+len(xs),:,D.perm] = R.predict_probs(model,xs)
            np.save(path,out)
            source.write_text(json.dumps(dict(model_sha256=fingerprint,model=str(MODEL),
                  split=split,panel=panel,n_windows=len(ws),shape=list(out.shape)),indent=2))
            print(split,panel,len(ws),f"{time.time()-start:.1f}s",flush=True)


def evaluate_variant(data, split, alpha, theta=.45, base_config=None):
    rows=[]
    for panel,z in data.items():
        new=load_candidate_prob(panel,split)
        pr=(1-alpha)*E.probs(z,(base_config or E.BASE)["weights"])+alpha*new
        ious=E.scores(z,pr>=theta)
        rows.extend(dict(panel=panel,day=int(d),T=int(t),date=str(date),iou=float(iou))
          for d,t,date,iou in zip(z["day"],z["T"],z["date"],ious))
    import pandas as pd
    return pd.DataFrame(rows)


def cmd_select(args):
    import pandas as pd
    data=E.load("validation")
    rows=[]
    bases={"v26":E.BASE,"v27_rules":json.loads((E.OUT/"selected.json").read_text())}
    for base,config in bases.items():
        for alpha in (0,.125,.25,.375,.5):
            for theta in (.4,.425,.45,.475,.5):
                df=evaluate_variant(data,"validation",alpha,theta,config)
                pm=df.groupby("panel").iou.mean()
                rows.append(dict(base=base,alpha=alpha,theta=theta,score=float(pm.mean()),**pm.to_dict()))
    ranked=pd.DataFrame(rows).sort_values("score",ascending=False)
    ranked.to_csv(E.OUT/"lovasz_selection_validation.csv",index=False)
    selected=ranked.iloc[0][["alpha","theta"]].to_dict()
    selected["base"]=str(ranked.iloc[0]["base"])
    selected["base_config"]=bases[selected["base"]]
    selected["reference_config"]=bases["v27_rules"]
    selected["model_sha256"]=model_sha256()
    (E.OUT/"lovasz_selected.json").write_text(json.dumps(selected,indent=2))
    print(ranked.head(15).to_string(index=False),flush=True)


def cmd_confirm(args):
    import pandas as pd
    selected=checked_selection()
    rows=[]
    for split in ("validation","private"):
        data=E.load(split)
        for name,alpha,theta,base_config in (("v26",0,.45,E.BASE),
                ("v27_rules",0,selected["reference_config"]["theta"],selected["reference_config"]),
                ("v27_lovasz",selected["alpha"],selected["theta"],selected["base_config"])):
            df=evaluate_variant(data,split,alpha,theta,base_config)
            df["model"],df["split"]=name,split
            rows.append(df)
    df=pd.concat(rows,ignore_index=True)
    df.to_csv(E.OUT/"lovasz_confirmed_windows.csv",index=False)
    pm=df.groupby(["split","model","panel"]).iou.mean().reset_index()
    pm.to_csv(E.OUT/"lovasz_confirmed_panels.csv",index=False)
    (E.OUT/"lovasz_confirmed_manifest.json").write_text(json.dumps(dict(config=selected,
          panels_sha256=E.file_sha256(E.OUT/"lovasz_confirmed_panels.csv")),indent=2))
    print(pm.pivot(index=["split","panel"],columns="model",values="iou").to_string(),flush=True)
    print(pm.groupby(["split","model"]).iou.mean().to_string(),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("train")
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--samples",type=int,default=6000)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--lr",type=float,default=2e-4)
    sub.add_parser("cache")
    sub.add_parser("select")
    sub.add_parser("confirm")
    args=parser.parse_args()
    globals()["cmd_"+args.cmd](args)
