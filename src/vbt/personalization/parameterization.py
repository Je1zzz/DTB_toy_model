"""Frozen graph parameterization for patient excitability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class GraphParameterization:
    population_x0: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    processed_weights: np.ndarray
    laplacian: np.ndarray

    @classmethod
    def from_connectome(cls, population_x0: np.ndarray, weights: np.ndarray, rank: int):
        raw = np.asarray(weights, dtype=float)
        mean = np.asarray(population_x0, dtype=float)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or mean.shape != (raw.shape[0],):
            raise ValueError("population_x0 and square weights are required")
        clipped = np.maximum(raw, 0.0); np.fill_diagonal(clipped, 0.0)
        symmetric = 0.5 * (clipped + clipped.T)
        radius = float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))
        processed = symmetric / (radius + 1e-8)
        degree = processed.sum(axis=1)
        inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        laplacian = np.eye(raw.shape[0]) - inv_sqrt[:, None] * processed * inv_sqrt[None, :]
        isolated = degree == 0
        laplacian[isolated, isolated] = 0.0
        values, vectors = np.linalg.eigh(laplacian)
        if rank < 1 or rank > mean.size:
            raise ValueError(f"rank must be in [1, {mean.size}]")
        return cls(mean, vectors[:, :rank], values[:rank], processed, laplacian)

    @property
    def rank(self): return self.basis.shape[1]
    def expand(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.rank,): raise ValueError(f"alpha must have shape ({self.rank},)")
        return self.population_x0 + self.basis @ alpha
    def project(self, x0):
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != self.population_x0.shape: raise ValueError("x0 shape mismatch")
        return self.basis.T @ (x0 - self.population_x0)
    def decompose(self, x0):
        """Return unique graph coefficients and orthogonal sparse residual."""
        values=np.asarray(x0,dtype=float); alpha=self.project(values)
        delta=values-self.population_x0-self.basis@alpha
        return alpha,delta
    def spectral_precision(self, roi_scale=0.30, gamma=1.0):
        scale=roi_scale*np.sqrt(self.population_x0.size/self.rank)
        denominator=max(float(self.eigenvalues[-1]),1e-8)
        return (1+gamma*self.eigenvalues/denominator)/(scale**2)
    def manifest(self):
        digest = lambda x: hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()
        return {"rank": self.rank, "weights_sha256": digest(self.processed_weights), "laplacian_sha256": digest(self.laplacian)}
