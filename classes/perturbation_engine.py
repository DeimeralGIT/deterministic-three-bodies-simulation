from __future__ import annotations

import numpy as np

from .body import Body


class PerturbationEngine:
    def __init__(self, mode: str = "A", noise_scale: float = 1e-4, seed: int = 7) -> None:
        self.mode = mode.upper()
        self.noise_scale = float(noise_scale)
        self.rng = np.random.default_rng(seed)
        self._last_magnitude = 0.0
        self._last_cancelled = 0.0

    @property
    def last_magnitude(self) -> float:
        return self._last_magnitude

    @property
    def last_cancelled(self) -> float:
        return self._last_cancelled

    def apply(self, bodies: list[Body]) -> None:
        if self.mode == "A":
            self._last_magnitude = 0.0
            self._last_cancelled = 0.0
            return
        if self.mode == "B":
            self._apply_random_noise(bodies)
            return
        if self.mode == "C":
            self._apply_symmetric_cancellation(bodies)
            return
        raise ValueError(f"Unsupported mode for single-path perturbation: {self.mode}")

    def branch_perturbations(self, n_branches: int, n_bodies: int, dim: int) -> np.ndarray:
        if self.mode != "D":
            return np.zeros((n_branches, n_bodies, dim), dtype=float)
        raw = self.rng.normal(0.0, self.noise_scale, size=(n_branches, n_bodies, dim))
        centered = raw - raw.mean(axis=0, keepdims=True)
        self._last_magnitude = float(np.mean(np.linalg.norm(centered.reshape(-1, dim), axis=1)))
        # Branch-centering enforces cancellation in expectation across branches.
        self._last_cancelled = 1.0
        return centered

    def _apply_random_noise(self, bodies: list[Body]) -> None:
        dim = len(bodies[0].position)
        vel_kicks = self.rng.normal(0.0, self.noise_scale, size=(len(bodies), dim))
        for i, b in enumerate(bodies):
            b.velocity += vel_kicks[i]
        self._last_magnitude = float(np.mean(np.linalg.norm(vel_kicks, axis=1)))
        self._last_cancelled = 0.0

    def _apply_symmetric_cancellation(self, bodies: list[Body]) -> None:
        dim = len(bodies[0].position)
        kicks = np.zeros((len(bodies), dim), dtype=float)

        # Pairwise equal-opposite kicks keep total injected momentum close to zero.
        for i in range(0, len(bodies) - 1, 2):
            eps = self.rng.normal(0.0, self.noise_scale, size=dim)
            kicks[i] += eps
            kicks[i + 1] -= eps

        if len(bodies) % 2 == 1:
            i = len(bodies) - 1
            eps = self.rng.normal(0.0, self.noise_scale, size=dim)
            kicks[i] += eps
            kicks -= eps / len(bodies)

        for i, b in enumerate(bodies):
            b.velocity += kicks[i]

        avg = kicks.mean(axis=0)
        residual = np.linalg.norm(avg)
        gross = np.mean(np.linalg.norm(kicks, axis=1)) + 1e-16
        self._last_magnitude = float(gross)
        self._last_cancelled = float(max(0.0, min(1.0, 1.0 - residual / gross)))
