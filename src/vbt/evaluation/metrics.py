import numpy as np
def auc(scores,truth):
    s=np.asarray(scores); t=np.asarray(truth,bool); p=s[t]; n=s[~t]
    return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())
def average_precision(scores,truth):
    """Tie-grouped AP, invariant to ROI ordering within equal-score groups."""
    s=np.asarray(scores,float); t=np.asarray(truth,bool)
    if s.shape != t.shape or s.ndim != 1: raise ValueError("scores/truth shape mismatch")
    positives=int(t.sum())
    if not positives: return float("nan")
    total=0; found=0; area=0.0
    for value in np.unique(s)[::-1]:
        group=s==value; size=int(group.sum()); group_positive=int(t[group].sum())
        total += size; found += group_positive
        area += group_positive * found / total
    return float(area/positives)
def rank_metrics(scores,truth):
    """Tie-aware ranks and fractional Recall@number-of-positives."""
    s=np.asarray(scores,float); t=np.asarray(truth,bool); k=int(t.sum())
    if s.shape != t.shape or s.ndim != 1 or not k: raise ValueError("truth must contain positives and match scores")
    positive_ranks=[]; consumed=0; selected_positive=0.0
    for value in np.unique(s)[::-1]:
        group=s==value; size=int(group.sum()); group_positive=int(t[group].sum())
        average_rank=consumed+(size+1)/2
        positive_ranks.extend([average_rank]*group_positive)
        take=min(max(k-consumed,0),size)
        selected_positive += take*group_positive/size
        consumed += size
    return {"first_ez_rank":float(min(positive_ranks)),"mean_ez_rank":float(np.mean(positive_ranks)),
            "oracle_k_recall":float(selected_positive/k)}

def meaningful_pair_concordance(estimate,truth,minimum_difference=0.05,prediction_tie=1e-8):
    estimate=np.asarray(estimate,float); truth=np.asarray(truth,float)
    if estimate.shape != truth.shape or estimate.ndim != 1: raise ValueError("estimate/truth shape mismatch")
    values=[]
    for i in range(truth.size):
        for j in range(i):
            difference=truth[i]-truth[j]
            if abs(difference) < minimum_difference: continue
            predicted=estimate[i]-estimate[j]
            values.append(.5 if abs(predicted)<prediction_tie else float(np.sign(predicted)==np.sign(difference)))
    return float(np.mean(values)) if values else float("nan")
