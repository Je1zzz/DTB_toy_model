import numpy as np
def auc(scores,truth):
    s=np.asarray(scores); t=np.asarray(truth,bool); p=s[t]; n=s[~t]
    return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())
def average_precision(scores,truth):
    order=np.argsort(scores)[::-1]; t=np.asarray(truth,bool)[order]; return float(np.mean(np.cumsum(t)[t]/(np.flatnonzero(t)+1))) if t.any() else float("nan")
def rank_metrics(scores,truth):
    order=np.argsort(scores)[::-1]; ranks=np.flatnonzero(np.asarray(truth,bool)[order])+1; k=len(ranks)
    return {"first_ez_rank":int(ranks.min()),"mean_ez_rank":float(ranks.mean()),"oracle_k_recall":float(np.asarray(truth,bool)[order[:k]].mean())}
