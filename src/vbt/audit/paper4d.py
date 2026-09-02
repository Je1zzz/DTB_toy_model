"""The four-state equation printed in Wang et al., Nature CS (2025).

This mathematical audit is deliberately not an executable pipeline profile.
``x0`` and ``b`` are explicit because the paper's listed defaults do not fully
specify them.
"""

from __future__ import annotations

import numpy as np


def paper4d_rhs(state, x0, stimulus, local_term, global_term, *, b):
    x, y, z, m = np.asarray(state, dtype=float)
    switch = np.heaviside(m - 1.8, 1.0)
    return np.asarray([
        y - x**3 + b * x**2 - z + 3.1 + stimulus + local_term + global_term,
        1.0 - 5.0 * x**2 - y,
        4.0 * (x - np.asarray(x0) + switch) - z,
        stimulus - m,
    ])
