import numpy as np
from vbt.evaluation.metrics import auc,average_precision,rank_metrics
def evaluate(scores,truth):
    value={"auroc":auc(scores,truth),"average_precision":average_precision(scores,truth),"prevalence":float(np.mean(truth))}
    value.update(rank_metrics(scores,truth)); return value
def permutation_p(scores,truth,n=1000,seed=0):
    rng=np.random.default_rng(seed); observed=evaluate(scores,truth); null=np.array([auc(scores,rng.permutation(truth)) for _ in range(n)])
    return {"auroc_p":float((1+np.sum(null>=observed["auroc"]))/(n+1)),"auroc_null_p95":float(np.quantile(null,.95))}
