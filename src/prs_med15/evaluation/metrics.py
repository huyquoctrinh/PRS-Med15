"""Per-action overlap metrics and patient-clustered confidence intervals."""
from __future__ import annotations
import numpy as np
import torch


def overlap_metrics(pred: torch.Tensor,target: torch.Tensor,threshold: float=0.0)->dict[str,float|str]:
    p=pred>threshold; t=target.bool(); inter=(p&t).sum().item(); ps=p.sum().item(); ts=t.sum().item()
    union=(p|t).sum().item()
    return {"dice":(2*inter)/(ps+ts) if ps+ts else 1.,"iou":inter/union if union else 1.,
            "dimension":"2d" if target.ndim<=2 or (target.ndim==3 and target.shape[0]==1) else "3d"}


def invalid_action_rate(valid: list[bool])->float: return 1-sum(valid)/len(valid) if valid else 0.


def patient_clustered_bootstrap(values: list[tuple[str,float]],samples:int=1000,seed:int=0,alpha:float=.05):
    groups={k:[] for k,_ in values}
    for k,v in values: groups[k].append(v)
    keys=list(groups); rng=np.random.default_rng(seed); draws=[]
    for _ in range(samples):
        sampled=rng.choice(keys,len(keys),replace=True); draws.append(np.mean([v for k in sampled for v in groups[k]]))
    return {"mean":float(np.mean([v for _,v in values])),"low":float(np.quantile(draws,alpha/2)),"high":float(np.quantile(draws,1-alpha/2))}
