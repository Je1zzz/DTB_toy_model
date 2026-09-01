import numpy as np
def reconstruct_weights(gain_prior_row):
    values=np.asarray(gain_prior_row,float); span=np.ptp(values)
    return (values-values.min())/span if span else np.zeros_like(values)
