from __future__ import annotations

from typing import Iterable
import numpy as np

from .body import Body


class PhysicsEngine:
    def __init__(self, gravitational_constant: float = 1.0, softening: float = 1e-5) -> None:
        self.G = gravitational_constant
        self.softening = softening

    def accelerations(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        n, dim = positions.shape
        acc = np.zeros((n, dim), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = positions[j] - positions[i]
                dist2 = float(np.dot(r, r) + self.softening * self.softening)
                inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
                acc[i] += self.G * masses[j] * r * inv_dist3
        return acc

    def set_accelerations(self, bodies: list[Body]) -> None:
        p, _, m = self._state_from_bodies(bodies)
        a = self.accelerations(p, m)
        for i, b in enumerate(bodies):
            b.acceleration = a[i]

    def rk4_step(self, bodies: list[Body], dt: float) -> None:
        p0, v0, m = self._state_from_bodies(bodies)

        def deriv(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return vel, self.accelerations(pos, m)

        k1_p, k1_v = deriv(p0, v0)
        k2_p, k2_v = deriv(p0 + 0.5 * dt * k1_p, v0 + 0.5 * dt * k1_v)
        k3_p, k3_v = deriv(p0 + 0.5 * dt * k2_p, v0 + 0.5 * dt * k2_v)
        k4_p, k4_v = deriv(p0 + dt * k3_p, v0 + dt * k3_v)

        p1 = p0 + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        v1 = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        a1 = self.accelerations(p1, m)
        self._write_state_to_bodies(bodies, p1, v1, a1)

    def velocity_verlet_step(self, bodies: list[Body], dt: float) -> None:
        p0, v0, m = self._state_from_bodies(bodies)
        a0 = self.accelerations(p0, m)

        p1 = p0 + v0 * dt + 0.5 * a0 * dt * dt
        a1 = self.accelerations(p1, m)
        v1 = v0 + 0.5 * (a0 + a1) * dt

        self._write_state_to_bodies(bodies, p1, v1, a1)

    def total_energy(self, bodies: Iterable[Body]) -> float:
        bodies_list = list(bodies)
        kinetic = 0.0
        potential = 0.0

        for b in bodies_list:
            kinetic += 0.5 * b.mass * float(np.dot(b.velocity, b.velocity))

        n = len(bodies_list)
        for i in range(n):
            for j in range(i + 1, n):
                r = bodies_list[j].position - bodies_list[i].position
                dist = np.sqrt(float(np.dot(r, r) + self.softening * self.softening))
                potential -= self.G * bodies_list[i].mass * bodies_list[j].mass / dist

        return kinetic + potential

    def total_momentum(self, bodies: Iterable[Body]) -> np.ndarray:
        p = None
        for b in bodies:
            if p is None:
                p = np.zeros_like(b.velocity)
            p += b.mass * b.velocity
        return p if p is not None else np.zeros(2, dtype=float)

    def choose_adaptive_dt(self, bodies: list[Body], base_dt: float, min_dt: float = 1e-5) -> float:
        # Keep integration stable near close encounters by shrinking dt as distance decreases.
        min_dist = np.inf
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                d = float(np.linalg.norm(bodies[j].position - bodies[i].position))
                if d < min_dist:
                    min_dist = d
        if not np.isfinite(min_dist):
            return base_dt
        scale = max(0.15, min(1.0, min_dist))
        return max(min_dt, base_dt * scale)

    @staticmethod
    def _state_from_bodies(bodies: list[Body]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.stack([b.position for b in bodies], axis=0)
        velocities = np.stack([b.velocity for b in bodies], axis=0)
        masses = np.array([b.mass for b in bodies], dtype=float)
        return positions, velocities, masses

    @staticmethod
    def _write_state_to_bodies(
        bodies: list[Body], positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray
    ) -> None:
        for i, b in enumerate(bodies):
            b.position = positions[i]
            b.velocity = velocities[i]
            b.acceleration = accelerations[i]
