from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Body:
    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def copy(self) -> "Body":
        return Body(
            name=self.name,
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
        )

    @staticmethod
    def from_values(name: str, mass: float, position: list[float], velocity: list[float]) -> "Body":
        return Body(
            name=name,
            mass=float(mass),
            position=np.array(position, dtype=float),
            velocity=np.array(velocity, dtype=float),
            acceleration=np.zeros(len(position), dtype=float),
        )
