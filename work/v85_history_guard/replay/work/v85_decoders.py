"""Experimental queue decoders; all selection uses probabilities, never horizon labels."""
import numpy as np

def ratio(p):
 f=np.asarray(p).ravel();o=np.argsort(-f,kind='stable');cs=np.cumsum(f[o]);den=np.arange(1,len(f)+1)+f.sum()-cs;k=int(np.argmax(np.divide(cs,den,out=np.zeros_like(cs),where=den>0)))+1 if f.sum()>0 else 0
 z=np.zeros(len(f),bool);z[o[:k]]=True;return z.reshape(p.shape)

def samples(p,ids,seed,constrained=False,K=20000):
 rng=np.random.default_rng(seed);_,inv=np.unique(ids,return_inverse=True);U=rng.random((K,inv.max()+1),dtype=np.float32)[:,inv];Y=U[:,None,:]<p[None].astype(np.float32)
 if constrained:
  A=Y[:,0];a=A.sum(1);i=(Y&A[:,None]).sum(2);u=(Y|A[:,None]).sum(2);i2=a+i[:,1:6].sum(1);u2=a+u[:,1:6].sum(1);i1=2*a+i[:,1:5].sum(1);u1=2*a+u[:,1:5].sum(1);ok=(u2>0)&(i2<=.9*u2)&(u1>0)&(i1>.9*u1)
 else:ok=Y.any((1,2))
 return Y[ok],float(ok.mean())

def utilities(c,Y):
 yy=Y.reshape(len(Y),-1);cc=c.ravel();i=yy[:,cc].sum(1);u=yy.sum(1)+cc.sum()-i
 return np.divide(i,u,out=np.ones(len(i),float),where=u>0)

def decode(base,Y,seed=0):
 """Optimize prefix sets keeping h1 fixed. Independent MC check gates conservative variant."""
 if len(Y)<100:return base.copy(),base.copy(),{'accepted_samples':len(Y),'reason':'too_few_samples'}
 rng=np.random.default_rng(seed);order_samples=rng.permutation(len(Y));cut=min(2048,len(Y)//2);train=Y[order_samples[:cut]];test=Y[order_samples[cut:cut+2048]]
 q=train.mean(0);free=np.arange(base.size).reshape(base.shape)[1:].ravel();rank=free[np.argsort(-q.ravel()[free],kind='stable')];flat=base.ravel();fixed=np.zeros(base.size,bool);fixed[:base.shape[1]]=flat[:base.shape[1]]
 maxk=min(len(rank),max(int(base.sum()*2+30),40));rank=rank[:maxk];yy=train.reshape(len(train),-1);pre=np.cumsum(yy[:,rank],axis=1,dtype=np.int32);pre=np.column_stack([np.zeros(len(train),int),pre]);fixed_i=yy[:,fixed].sum(1);I=pre+fixed_i[:,None];U=yy.sum(1)[:,None]+fixed.sum()+np.arange(maxk+1)[None]-I;ui=np.divide(I,U,out=np.ones(I.shape,float),where=U>0);mean=ui.mean(0);best=int(np.argmax(mean));z=fixed.copy();z[rank[:best]]=True;z=z.reshape(base.shape)
 ub=utilities(base,train).mean()
 if mean[best]<=ub+1e-6:z=base.copy()
 # Conservative candidate: at most 10% of base positives (minimum two) flips; confirm expected gain on separate samples.
 candidates=[];budget=max(2,int(np.ceil(base.sum()*.1)))
 for k in range(maxk+1):
  cc=fixed.copy();cc[rank[:k]]=True;flips=int((cc!=flat).sum())
  if 0<flips<=budget:candidates.append((float(mean[k]),k))
 robust=base.copy();lower=0.;gain=0.
 if candidates:
  _,k=max(candidates);cc=fixed.copy();cc[rank[:k]]=True;cc=cc.reshape(base.shape);diff=utilities(cc,test)-utilities(base,test);gain=float(diff.mean());lower=gain-2.58*float(diff.std(ddof=1))/np.sqrt(len(diff))
  if lower>0.0001:robust=cc
 return z,robust,{'accepted_samples':len(Y),'joint_flips':int((z!=base).sum()),'robust_flips':int((robust!=base).sum()),'mc_gain_check':gain,'mc_gain_lower':lower}

def iou(pred,y,m):
 a=pred&m;b=y&m;u=(a|b).sum();return float((a&b).sum()/u) if u else 1.
