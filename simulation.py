from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

from classes import Body, DeterminismAnalyzer, PerturbationEngine, PhysicsEngine, Renderer


DEFAULT_ORBIT_LIMIT_X = 12.0
DEFAULT_ORBIT_LIMIT_Y = 8.0
DEFAULT_ORBIT_LIMIT_Z = 8.0
DEFAULT_START_PADDING = 3.0
INITIAL_VELOCITY_MULTIPLIER = 2.0
MAX_BODIES = 8
BODY_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def mass_range_for_box(orbit_limit_x: float, orbit_limit_y: float, orbit_limit_z: float) -> tuple[float, float]:
    """Superlinear mass range from sandbox volume with per-side limits clamped to [6, 18].

    Anchor points:
    - 12x8x8 -> 4..5
    - 12x12x12 -> 8..9
    """
    x = max(6.0, min(18.0, float(orbit_limit_x)))
    y = max(6.0, min(18.0, float(orbit_limit_y)))
    z = max(6.0, min(18.0, float(orbit_limit_z)))
    volume = x * y * z

    # t=0 at V=768 (12x8x8), t=1 at V=1728 (12x12x12).
    # Superlinear (>1 exponent) to grow faster at larger sandbox volumes.
    t = max(0.0, (volume - 768.0) / 960.0)
    min_mass = 4.0 + 4.0 * (t ** 1.2)
    max_mass = min_mass + 1.0
    return float(min_mass), float(max_mass)


@dataclass
class RunData:
    time: np.ndarray
    trajectories: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    body_names: list[str]
    energy: np.ndarray
    momentum_mag: np.ndarray
    divergence: np.ndarray
    perturbation_mag: np.ndarray
    cancellation: np.ndarray
    branch_divergence: np.ndarray | None


@dataclass
class LiveSnapshot:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    body_names: list[str]
    time: float
    step_index: int
    finished: bool


class LiveSimulation:
    def __init__(
        self,
        args: argparse.Namespace,
        rng: np.random.Generator | None = None,
        initial_override: list[Body] | None = None,
        apply_velocity_multiplier: bool = True,
        enforce_start_clamp: bool = True,
        enforce_bound: bool = True,
    ) -> None:
        mode = args.mode.upper()
        if mode not in {"A", "B", "C", "D"}:
            raise ValueError("mode must be one of A, B, C, D")

        self.args = args
        self.mode = mode
        self.physics = PhysicsEngine(gravitational_constant=args.gravity, softening=args.softening)
        self.perturb = PerturbationEngine(mode=mode, noise_scale=args.noise_scale, seed=args.seed)

        initial = clone_bodies(initial_override) if initial_override is not None else build_initial_conditions(args, rng)
        if apply_velocity_multiplier:
            scale_initial_velocities(initial, INITIAL_VELOCITY_MULTIPLIER)
        apply_com_frame(initial)
        if enforce_start_clamp:
            clamp_start_positions_to_inner_window(
                initial,
                args.orbit_limit_x,
                args.orbit_limit_y,
                args.orbit_limit_z,
                args.start_padding,
            )
        if enforce_bound and (args.bound or rng is not None):
            rescale_to_bound(initial, self.physics)

        self.baseline = clone_bodies(initial)
        self.integrator = select_integrator(self.physics, args.integrator)
        self.current_step = 0
        self.current_time = 0.0
        self.masses = np.array([b.mass for b in initial], dtype=float)
        self.body_names = [b.name for b in initial]

        if mode == "D":
            self.branches = [clone_bodies(initial) for _ in range(args.branches)]
            self.main = None
        else:
            self.main = clone_bodies(initial)
            self.branches = None
            seed_offset = np.zeros_like(self.main[0].velocity)
            seed_values = np.array([1e-9, -1e-9, 1e-9], dtype=float)
            seed_offset[: min(len(seed_offset), len(seed_values))] = seed_values[: min(len(seed_offset), len(seed_values))]
            self.main[0].velocity = self.main[0].velocity + seed_offset

    def snapshot(self) -> LiveSnapshot:
        positions, velocities = self._display_state()
        return LiveSnapshot(
            positions=positions,
            velocities=velocities,
            masses=self.masses.copy(),
            body_names=list(self.body_names),
            time=self.current_time,
            step_index=self.current_step,
            finished=False,
        )

    def step(self, step_count: int = 1) -> LiveSnapshot:
        for _ in range(max(0, int(step_count))):
            dt_eff = self.physics.choose_adaptive_dt(self.baseline, self.args.dt) if self.args.adaptive else self.args.dt
            self.current_time += dt_eff
            self.integrator(self.baseline, dt_eff)

            if self.mode == "D":
                assert self.branches is not None
                for branch in self.branches:
                    self.integrator(branch, dt_eff)

                n_bodies = len(self.branches[0])
                dim = len(self.branches[0][0].position)
                kicks = self.perturb.branch_perturbations(self.args.branches, n_bodies, dim)
                for b_idx, branch in enumerate(self.branches):
                    for i in range(n_bodies):
                        branch[i].velocity += kicks[b_idx, i]
            else:
                assert self.main is not None
                self.integrator(self.main, dt_eff)
                self.perturb.apply(self.main)

            self.current_step += 1

        return self.snapshot()

    def _display_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "D":
            assert self.branches is not None
            branch_positions = np.stack([np.stack([b.position for b in br], axis=0) for br in self.branches], axis=0)
            branch_velocities = np.stack([np.stack([b.velocity for b in br], axis=0) for br in self.branches], axis=0)
            return np.mean(branch_positions, axis=0), np.mean(branch_velocities, axis=0)

        assert self.main is not None
        return (
            np.stack([b.position for b in self.main], axis=0),
            np.stack([b.velocity for b in self.main], axis=0),
        )


def create_live_simulation(
    args: argparse.Namespace,
    rng: np.random.Generator | None = None,
    initial_override: list[Body] | None = None,
    apply_velocity_multiplier: bool = True,
    enforce_start_clamp: bool = True,
    enforce_bound: bool = True,
) -> LiveSimulation:
    return LiveSimulation(
        args=args,
        rng=rng,
        initial_override=initial_override,
        apply_velocity_multiplier=apply_velocity_multiplier,
        enforce_start_clamp=enforce_start_clamp,
        enforce_bound=enforce_bound,
    )


def clone_bodies(bodies: list[Body]) -> list[Body]:
    return [b.copy() for b in bodies]


def figure_eight_initial_conditions() -> list[Body]:
    return [
        Body.from_values("A", 1.0, [-0.97000436, 0.24308753, 0.0], [0.4662036850, 0.4323657300, 0.0]),
        Body.from_values("B", 1.0, [0.97000436, -0.24308753, 0.0], [0.4662036850, 0.4323657300, 0.0]),
        Body.from_values("C", 1.0, [0.0, 0.0, 0.0], [-0.93240737, -0.86473146, 0.0]),
    ]


def random_initial_conditions(
    rng: np.random.Generator,
    orbit_limit_x: float = DEFAULT_ORBIT_LIMIT_X,
    orbit_limit_y: float = DEFAULT_ORBIT_LIMIT_Y,
    orbit_limit_z: float = DEFAULT_ORBIT_LIMIT_Z,
    start_padding: float = DEFAULT_START_PADDING,
    body_count: int = 3,
) -> list[Body]:
    """Randomise masses, positions, and velocities within a padded inner start region."""
    n = max(1, min(MAX_BODIES, int(body_count)))
    mass_min, mass_max = mass_range_for_box(orbit_limit_x, orbit_limit_y, orbit_limit_z)
    masses = rng.uniform(mass_min, mass_max, size=n)
    inner_limit_x = max(0.5, orbit_limit_x - max(0.1, start_padding))
    inner_limit_y = max(0.5, orbit_limit_y - max(0.1, start_padding))
    inner_limit_z = max(0.5, orbit_limit_z - max(0.1, start_padding))
    inner_limit = min(inner_limit_x, inner_limit_y, inner_limit_z)
    min_radius = max(0.2, 0.10 * inner_limit)

    # Place bodies within a padded inner 3D volume with room to move in all directions.
    raw_positions = np.zeros((n, 3), dtype=float)
    min_sep = max(0.35, 0.16 * inner_limit)
    for i in range(n):
        placed = False
        for _ in range(200):
            direction = rng.normal(0.0, 1.0, size=3)
            direction /= float(np.linalg.norm(direction) + 1e-12)
            radius = float(rng.uniform(min_radius, 0.65 * inner_limit))
            candidate = direction * radius
            if i == 0:
                raw_positions[i] = candidate
                placed = True
                break
            dists = np.linalg.norm(raw_positions[:i] - candidate[None, :], axis=1)
            if np.all(dists > min_sep):
                raw_positions[i] = candidate
                placed = True
                break
        if not placed:
            raw_positions[i] = rng.uniform(
                low=[-0.65 * inner_limit_x, -0.65 * inner_limit_y, -0.65 * inner_limit_z],
                high=[0.65 * inner_limit_x, 0.65 * inner_limit_y, 0.65 * inner_limit_z],
                size=3,
            )

    # Respect anisotropic window bounds (x/y/z can differ).
    scale_limits = [1.0]
    max_abs_x = np.max(np.abs(raw_positions[:, 0]))
    max_abs_y = np.max(np.abs(raw_positions[:, 1]))
    max_abs_z = np.max(np.abs(raw_positions[:, 2]))
    if max_abs_x > inner_limit_x:
        scale_limits.append(float(inner_limit_x / max_abs_x))
    if max_abs_y > inner_limit_y:
        scale_limits.append(float(inner_limit_y / max_abs_y))
    if max_abs_z > inner_limit_z:
        scale_limits.append(float(inner_limit_z / max_abs_z))
    positions_arr = raw_positions * min(scale_limits)

    # Double start speeds to reduce early collapse/collision.
    speed_scale = rng.uniform(0.56, 1.44)
    velocities_arr = np.zeros((n, 3), dtype=float)
    for i in range(n):
        r = positions_arr[i]
        trial = rng.normal(0.0, 1.0, size=3)
        tangent = np.cross(r, trial)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm < 1e-10:
            tangent = np.cross(r, np.array([1.0, 0.0, 0.0], dtype=float))
            tnorm = float(np.linalg.norm(tangent))
        tangent /= (tnorm + 1e-12)
        velocities_arr[i] = tangent * speed_scale * rng.uniform(0.7, 1.3)

    positions = positions_arr.tolist()
    velocities = velocities_arr.tolist()
    return [Body.from_values(BODY_NAMES[i], float(masses[i]), positions[i], velocities[i]) for i in range(n)]


def parse_triplet_floats(raw: str, item_name: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if len(values) != 3:
        raise ValueError(f"{item_name} must contain exactly 3 comma-separated values")
    return [float(v) for v in values]


def parse_vector_list(raw: str, item_name: str, vec_dim: int = 3) -> list[list[float]]:
    groups = [g.strip() for g in raw.split(";") if g.strip()]
    if len(groups) != 3:
        raise ValueError(f"{item_name} must contain exactly 3 vectors separated by ';'")
    vectors: list[list[float]] = []
    for g in groups:
        comps = [x.strip() for x in g.split(",") if x.strip()]
        if vec_dim == 3 and len(comps) == 2:
            vectors.append([float(comps[0]), float(comps[1]), 0.0])
            continue
        if len(comps) != vec_dim:
            raise ValueError(f"Each {item_name} vector must have exactly {vec_dim} comma-separated components")
        vectors.append([float(x) for x in comps])
    return vectors


def build_initial_conditions(args: argparse.Namespace, rng: np.random.Generator | None = None) -> list[Body]:
    # If no custom args and randomise is requested (or is the default), generate randomly.
    if args.masses is None and args.positions is None and args.velocities is None:
        if rng is not None:
            return random_initial_conditions(
                rng,
                orbit_limit_x=args.orbit_limit_x,
                orbit_limit_y=args.orbit_limit_y,
                orbit_limit_z=args.orbit_limit_z,
                start_padding=args.start_padding,
                body_count=args.body_count,
            )
        return figure_eight_initial_conditions()

    if args.masses is None or args.positions is None or args.velocities is None:
        raise ValueError("When customizing initial conditions, provide --masses, --positions, and --velocities together")

    masses = [v.strip() for v in args.masses.split(",") if v.strip()]
    positions = parse_vector_list(args.positions, "positions", vec_dim=3)
    velocities = parse_vector_list(args.velocities, "velocities", vec_dim=3)
    if not (len(masses) == len(positions) == len(velocities)):
        raise ValueError("masses/positions/velocities must describe the same number of bodies")
    if len(masses) > MAX_BODIES:
        raise ValueError(f"At most {MAX_BODIES} bodies are supported")
    return [
        Body.from_values(BODY_NAMES[i], float(masses[i]), positions[i], velocities[i])
        for i in range(len(masses))
    ]


def solar_system_initial_conditions(
    gravity: float,
    orbit_limit_x: float,
    orbit_limit_y: float,
    orbit_limit_z: float,
    start_padding: float,
) -> list[Body]:
    # Normalized units (AU-like distance, year-like time): Sun + first 7 planets for max 8 bodies.
    # Distances are remapped nonlinearly for clearer visualization while preserving orbital ordering.
    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus"]
    masses = [1.0, 1.66e-7, 2.45e-6, 3.00e-6, 3.23e-7, 9.54e-4, 2.86e-4, 4.37e-5]
    radii = [0.0, 0.39, 0.72, 1.00, 1.52, 5.20, 9.58, 19.20]
    phases = [0.0, 0.3, 1.2, 2.0, 2.7, 3.5, 4.2, 5.0]

    inner_x = max(0.5, orbit_limit_x - max(0.1, start_padding))
    inner_y = max(0.5, orbit_limit_y - max(0.1, start_padding))
    inner_z = max(0.5, orbit_limit_z - max(0.1, start_padding))
    max_target_radius = 0.95 * min(inner_x, inner_y, inner_z)
    max_input_radius = max(radii)
    radial_exponent = 0.65

    bodies: list[Body] = [Body.from_values(names[0], masses[0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    g_eff = max(1e-9, float(gravity))
    for i in range(1, len(names)):
        r_norm = radii[i] / max_input_radius
        r = max_target_radius * (r_norm ** radial_exponent)
        th = phases[i]
        x, y = r * float(np.cos(th)), r * float(np.sin(th))
        v = float(np.sqrt(g_eff * masses[0] / max(r, 1e-9)))
        vx, vy = -v * float(np.sin(th)), v * float(np.cos(th))
        bodies.append(Body.from_values(names[i], masses[i], [x, y, 0.0], [vx, vy, 0.0]))
    return bodies


def add_random_body_separate(
    existing: list[Body],
    rng: np.random.Generator,
    orbit_limit_x: float,
    orbit_limit_y: float,
    orbit_limit_z: float,
    start_padding: float,
) -> Body:
    inner_x = max(0.5, orbit_limit_x - max(0.1, start_padding))
    inner_y = max(0.5, orbit_limit_y - max(0.1, start_padding))
    inner_z = max(0.5, orbit_limit_z - max(0.1, start_padding))
    inner = min(inner_x, inner_y, inner_z)
    min_sep = max(0.35, 0.16 * inner)

    center = np.mean(np.stack([b.position for b in existing], axis=0), axis=0) if existing else np.zeros(3, dtype=float)
    n = max(1, len(existing) + 1)
    mass_min, mass_max = mass_range_for_box(orbit_limit_x, orbit_limit_y, orbit_limit_z)
    mass = float(rng.uniform(mass_min, mass_max))
    name = BODY_NAMES[min(len(existing), MAX_BODIES - 1)]

    pos = np.zeros(3, dtype=float)
    for _ in range(250):
        pos = rng.uniform(low=[-inner_x, -inner_y, -inner_z], high=[inner_x, inner_y, inner_z], size=3)
        if not existing:
            break
        dists = np.linalg.norm(np.stack([b.position for b in existing], axis=0) - pos[None, :], axis=1)
        if np.all(dists > min_sep):
            break

    rel = pos - center
    trial = rng.normal(0.0, 1.0, size=3)
    tangent = np.cross(rel, trial)
    tnorm = float(np.linalg.norm(tangent))
    if tnorm < 1e-10:
        tangent = np.cross(rel, np.array([1.0, 0.0, 0.0], dtype=float))
        tnorm = float(np.linalg.norm(tangent))
    tangent /= (tnorm + 1e-12)
    speed = float(rng.uniform(0.56, 1.44) * rng.uniform(0.7, 1.3))
    vel = tangent * speed

    return Body.from_values(name, mass, pos.tolist(), vel.tolist())


def apply_com_frame(bodies: list[Body]) -> None:
    """Subtract centre-of-mass velocity so the system stays on screen."""
    total_mass = sum(b.mass for b in bodies)
    com_vel = sum(b.mass * b.velocity for b in bodies) / total_mass  # type: ignore[assignment]
    com_pos = sum(b.mass * b.position for b in bodies) / total_mass  # type: ignore[assignment]
    for b in bodies:
        b.velocity = b.velocity - com_vel
        b.position = b.position - com_pos


def rescale_to_bound(bodies: list[Body], physics: PhysicsEngine, margin: float = 0.5) -> None:
    """Scale down all velocities until total energy is negative (bound orbit)."""
    for _ in range(200):
        e = physics.total_energy(bodies)
        if e < 0:
            return
        for b in bodies:
            b.velocity *= (1.0 - margin * 0.05)


def scale_initial_velocities(bodies: list[Body], factor: float) -> None:
    for b in bodies:
        b.velocity *= factor


def clamp_start_positions_to_inner_window(
    bodies: list[Body], orbit_limit_x: float, orbit_limit_y: float, orbit_limit_z: float, start_padding: float
) -> None:
    """Keep starting positions inside a padded window: [-L+pad, L-pad] for x, y, and z."""
    inner_limit_x = max(0.5, orbit_limit_x - max(0.1, start_padding))
    inner_limit_y = max(0.5, orbit_limit_y - max(0.1, start_padding))
    inner_limit_z = max(0.5, orbit_limit_z - max(0.1, start_padding))

    max_abs_x = max(abs(float(b.position[0])) for b in bodies)
    max_abs_y = max(abs(float(b.position[1])) for b in bodies)
    max_abs_z = max(abs(float(b.position[2])) for b in bodies)
    if max_abs_x <= inner_limit_x and max_abs_y <= inner_limit_y and max_abs_z <= inner_limit_z:
        return

    scale_x = inner_limit_x / max_abs_x if max_abs_x > 0 else 1.0
    scale_y = inner_limit_y / max_abs_y if max_abs_y > 0 else 1.0
    scale_z = inner_limit_z / max_abs_z if max_abs_z > 0 else 1.0
    scale = min(1.0, scale_x, scale_y, scale_z)
    for b in bodies:
        b.position *= scale


def rms_divergence(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def select_integrator(engine: PhysicsEngine, name: str) -> Callable[[list[Body], float], None]:
    key = name.lower()
    if key == "rk4":
        return engine.rk4_step
    if key in {"verlet", "velocity-verlet", "velocity_verlet"}:
        return engine.velocity_verlet_step
    raise ValueError(f"Unsupported integrator: {name}")


def run_simulation(
    args: argparse.Namespace,
    rng: np.random.Generator | None = None,
    initial_override: list[Body] | None = None,
    apply_velocity_multiplier: bool = True,
    enforce_start_clamp: bool = True,
    enforce_bound: bool = True,
) -> RunData:
    mode = args.mode.upper()
    if mode not in {"A", "B", "C", "D"}:
        raise ValueError("mode must be one of A, B, C, D")

    if args.speed <= 0.0:
        raise ValueError("--speed must be > 0")

    physics = PhysicsEngine(gravitational_constant=args.gravity, softening=args.softening)
    perturb = PerturbationEngine(mode=mode, noise_scale=args.noise_scale, seed=args.seed)

    initial = clone_bodies(initial_override) if initial_override is not None else build_initial_conditions(args, rng)
    if apply_velocity_multiplier:
        scale_initial_velocities(initial, INITIAL_VELOCITY_MULTIPLIER)
    # Always simulate in the centre-of-mass frame so the system stays centred.
    apply_com_frame(initial)
    # Enforce padded inner start area so starts are always in-bounds for the orbit window.
    if enforce_start_clamp:
        clamp_start_positions_to_inner_window(
            initial,
            args.orbit_limit_x,
            args.orbit_limit_y,
            args.orbit_limit_z,
            args.start_padding,
        )
    # Always enforce bound orbit when using random ICs; optional otherwise.
    if enforce_bound and (args.bound or rng is not None):
        rescale_to_bound(initial, physics)
    baseline = clone_bodies(initial)
    integrator = select_integrator(physics, args.integrator)

    steps = int(args.steps)
    n_bodies = len(initial)
    dim = len(initial[0].position)

    time = np.zeros(steps + 1, dtype=float)
    trajectories = np.zeros((steps + 1, n_bodies, dim), dtype=float)
    velocities = np.zeros((steps + 1, n_bodies, dim), dtype=float)
    energy = np.zeros(steps + 1, dtype=float)
    momentum_mag = np.zeros(steps + 1, dtype=float)
    divergence = np.zeros(steps + 1, dtype=float)
    perturbation_mag = np.zeros(steps + 1, dtype=float)
    cancellation = np.zeros(steps + 1, dtype=float)

    branch_divergence = np.zeros(steps + 1, dtype=float) if mode == "D" else None

    if mode == "D":
        branches = [clone_bodies(initial) for _ in range(args.branches)]
    else:
        main = clone_bodies(initial)
        # Tiny seed offset lets us quantify sensitivity even in pure classical mode.
        seed_offset = np.zeros_like(main[0].velocity)
        seed_values = np.array([1e-9, -1e-9, 1e-9], dtype=float)
        seed_offset[: min(len(seed_offset), len(seed_values))] = seed_values[: min(len(seed_offset), len(seed_values))]
        main[0].velocity = main[0].velocity + seed_offset

    def stash(k: int, ref_bodies: list[Body], energy_source: list[Body], psource: list[Body], div_value: float) -> None:
        trajectories[k] = np.stack([b.position for b in ref_bodies], axis=0)
        velocities[k] = np.stack([b.velocity for b in ref_bodies], axis=0)
        energy[k] = physics.total_energy(energy_source)
        momentum_mag[k] = float(np.linalg.norm(physics.total_momentum(psource)))
        divergence[k] = div_value
        perturbation_mag[k] = perturb.last_magnitude
        cancellation[k] = perturb.last_cancelled

    if mode == "D":
        mean_pos = np.mean(np.stack([np.stack([b.position for b in branch], axis=0) for branch in branches], axis=0), axis=0)
        pseudo = clone_bodies(initial)
        for i, b in enumerate(pseudo):
            b.position = mean_pos[i]
        stash(0, pseudo, baseline, pseudo, rms_divergence(mean_pos, np.stack([b.position for b in baseline], axis=0)))
        branch_divergence[0] = divergence[0]
    else:
        stash(0, main, main, main, rms_divergence(
            np.stack([b.position for b in main], axis=0),
            np.stack([b.position for b in baseline], axis=0),
        ))

    t = 0.0
    for step in range(1, steps + 1):
        dt_eff = physics.choose_adaptive_dt(baseline, args.dt) if args.adaptive else args.dt
        t += dt_eff
        time[step] = t

        integrator(baseline, dt_eff)

        if mode == "D":
            for branch in branches:
                integrator(branch, dt_eff)

            kicks = perturb.branch_perturbations(args.branches, n_bodies, dim)
            for b_idx, branch in enumerate(branches):
                for i in range(n_bodies):
                    branch[i].velocity += kicks[b_idx, i]

            branch_positions = np.stack([np.stack([b.position for b in br], axis=0) for br in branches], axis=0)
            branch_velocities = np.stack([np.stack([b.velocity for b in br], axis=0) for br in branches], axis=0)

            mean_pos = np.mean(branch_positions, axis=0)
            mean_vel = np.mean(branch_velocities, axis=0)

            pseudo = clone_bodies(initial)
            for i, b in enumerate(pseudo):
                b.position = mean_pos[i]
                b.velocity = mean_vel[i]

            baseline_pos = np.stack([b.position for b in baseline], axis=0)
            div_main = rms_divergence(mean_pos, baseline_pos)
            div_branches = np.mean([rms_divergence(branch_positions[j], baseline_pos) for j in range(args.branches)])
            branch_divergence[step] = float(div_branches)

            stash(step, pseudo, baseline, pseudo, div_main)
        else:
            integrator(main, dt_eff)
            perturb.apply(main)
            baseline_pos = np.stack([b.position for b in baseline], axis=0)
            main_pos = np.stack([b.position for b in main], axis=0)
            stash(step, main, main, main, rms_divergence(main_pos, baseline_pos))

    return RunData(
        time=time,
        trajectories=trajectories,
        velocities=velocities,
        masses=np.array([b.mass for b in initial], dtype=float),
        body_names=[b.name for b in initial],
        energy=energy,
        momentum_mag=momentum_mag,
        divergence=divergence,
        perturbation_mag=perturbation_mag,
        cancellation=cancellation,
        branch_divergence=branch_divergence,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-body deterministic/perturbation simulation")
    parser.add_argument("--mode", type=str, default="A", help="A|B|C|D")
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--integrator", type=str, default="rk4", help="rk4|verlet")
    parser.add_argument("--noise-scale", type=float, default=2e-5)
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravitational constant strength")
    parser.add_argument("--softening", type=float, default=1e-5, help="Softening parameter (avoids singularities)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (example: 3.0)")
    parser.add_argument("--bound", action="store_true", help="Rescale velocities until total energy is negative (bound orbit)")
    parser.add_argument("--random", action="store_true", help="Randomise initial conditions each run (default when no --masses/positions/velocities given)")
    parser.add_argument("--escape-radius", type=float, default=0.0, help="Optional radial restart threshold; 0 disables radial restart")
    parser.add_argument("--adaptive", action="store_true", help="Use simple adaptive timestep")
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib display")
    parser.add_argument("--static", action="store_true", help="Disable orbit animation and render static trajectory")
    parser.add_argument("--frame-stride", type=int, default=4, help="Base frame stride for animation")
    parser.add_argument("--frame-skip", type=int, default=2, help="Render every Nth frame (default 2 for 50% frame skipping)")
    parser.add_argument("--interval-ms", type=int, default=25, help="Base animation frame interval in milliseconds")
    parser.add_argument("--body-count", type=int, default=3, help="Number of bodies for random initial conditions (1..8)")
    parser.add_argument("--orbit-limit-x", type=float, default=DEFAULT_ORBIT_LIMIT_X, help="Orbit plot x-axis limit Lx, shown as x in [-Lx, Lx]")
    parser.add_argument("--orbit-limit-y", type=float, default=DEFAULT_ORBIT_LIMIT_Y, help="Orbit plot y-axis limit Ly, shown as y in [-Ly, Ly]")
    parser.add_argument("--orbit-limit-z", type=float, default=DEFAULT_ORBIT_LIMIT_Z, help="Orbit plot z-axis limit Lz, shown as z in [-Lz, Lz]")
    parser.add_argument("--start-padding", type=float, default=DEFAULT_START_PADDING, help="Padding between start positions and orbit window edge")
    parser.add_argument("--masses", type=str, default=None, help="Three masses: m1,m2,m3")
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Three 3D positions: x1,y1,z1;x2,y2,z2;x3,y3,z3",
    )
    parser.add_argument(
        "--velocities",
        type=str,
        default=None,
        help="Three 3D velocities: vx1,vy1,vz1;vx2,vy2,vz2;vx3,vy3,vz3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.orbit_limit_x = max(6.0, min(16.0, float(args.orbit_limit_x)))
    args.orbit_limit_y = max(6.0, min(16.0, float(args.orbit_limit_y)))
    args.orbit_limit_z = max(6.0, min(16.0, float(args.orbit_limit_z)))
    args.body_count = max(1, min(MAX_BODIES, int(args.body_count)))
    use_random = args.random or (args.masses is None and args.positions is None and args.velocities is None)
    dashboard_rng = np.random.default_rng(args.seed)

    if use_random:
        dashboard_bodies = random_initial_conditions(
            dashboard_rng,
            orbit_limit_x=args.orbit_limit_x,
            orbit_limit_y=args.orbit_limit_y,
            orbit_limit_z=args.orbit_limit_z,
            start_padding=args.start_padding,
            body_count=args.body_count,
        )
    else:
        dashboard_bodies = build_initial_conditions(args, None)

    solar_base = solar_system_initial_conditions(
        gravity=args.gravity,
        orbit_limit_x=args.orbit_limit_x,
        orbit_limit_y=args.orbit_limit_y,
        orbit_limit_z=args.orbit_limit_z,
        start_padding=args.start_padding,
    )

    dashboard_in_solar_mode = False

    if args.no_show or args.static:
        run = run_simulation(
            args,
            None,
            initial_override=dashboard_bodies,
            apply_velocity_multiplier=True,
            enforce_start_clamp=True,
            enforce_bound=True,
        )

        analyzer = DeterminismAnalyzer()
        summary = analyzer.summarize(
            mode=args.mode.upper(),
            divergence=run.divergence,
            energy=run.energy,
            momentum_mag=run.momentum_mag,
            perturb_mag=run.perturbation_mag,
            cancellation=run.cancellation,
        )

        print("=== Numerical Metrics ===")
        print(f"Mode: {summary.mode}")
        print(f"G={args.gravity:.2f}, softening={args.softening:.6f}, playback_speed={args.speed:.1f}x")
        print(f"Max divergence: {summary.max_divergence:.6e}")
        print(f"Final divergence: {summary.final_divergence:.6e}")
        print(f"Relative energy drift (max): {summary.relative_energy_drift:.6e}")
        print(f"Max momentum drift: {summary.max_momentum_drift:.6e}")
        print(f"Mean perturbation magnitude: {summary.mean_perturbation:.6e}")
        print(f"Mean cancellation ratio: {summary.mean_cancellation:.4f}")
        print(f"Chaos amplification estimate: {summary.chaos_amplification:.6e}")
        print()
        print(analyzer.philosophical_report(summary))

    if not args.no_show:
        title = f"Three-Body Simulation | Mode {args.mode.upper()} | {args.integrator.upper()}"
        effective_stride = max(1, int(round(args.frame_stride * args.speed)))
        effective_interval = max(1, int(args.interval_ms / args.speed))

        if args.static:
            Renderer().plot_results(
                trajectories=run.trajectories,
                velocities=run.velocities,
                masses=run.masses,
                body_names=run.body_names,
                time=run.time,
                energy=run.energy,
                divergence=run.divergence,
                perturb_mag=run.perturbation_mag,
                branch_divergence=run.branch_divergence,
                title=title,
                animate=False,
                frame_stride=effective_stride,
                frame_skip=max(1, int(args.frame_skip)),
                interval_ms=effective_interval,
                escape_radius=args.escape_radius,
                orbit_limit_x=args.orbit_limit_x,
                orbit_limit_y=args.orbit_limit_y,
                orbit_limit_z=args.orbit_limit_z,
                refresh_fn=None,
            )
            return

        live_sim = create_live_simulation(
            args,
            None,
            initial_override=dashboard_bodies,
            apply_velocity_multiplier=True,
            enforce_start_clamp=True,
            enforce_bound=True,
        )

        def refresh_fn(
            requested_body_count: int | None = None,
            load_solar: bool = False,
            randomize_existing: bool = False,
        ) -> LiveSimulation:
            nonlocal dashboard_bodies, dashboard_in_solar_mode

            if load_solar:
                dashboard_bodies = clone_bodies(solar_base)
                dashboard_in_solar_mode = True

            target = len(dashboard_bodies) if requested_body_count is None else max(1, min(MAX_BODIES, int(requested_body_count)))

            # If user changes body count after solar preset, switch back to randomized bodies.
            if (not load_solar) and dashboard_in_solar_mode and requested_body_count is not None and target != len(dashboard_bodies):
                dashboard_bodies = random_initial_conditions(
                    dashboard_rng,
                    orbit_limit_x=args.orbit_limit_x,
                    orbit_limit_y=args.orbit_limit_y,
                    orbit_limit_z=args.orbit_limit_z,
                    start_padding=args.start_padding,
                    body_count=target,
                )
                dashboard_in_solar_mode = False

            # Manual refresh in non-solar mode should produce a new random configuration.
            if randomize_existing and not dashboard_in_solar_mode and not load_solar:
                dashboard_bodies = random_initial_conditions(
                    dashboard_rng,
                    orbit_limit_x=args.orbit_limit_x,
                    orbit_limit_y=args.orbit_limit_y,
                    orbit_limit_z=args.orbit_limit_z,
                    start_padding=args.start_padding,
                    body_count=target,
                )

            if target < len(dashboard_bodies):
                dashboard_bodies = dashboard_bodies[:target]
            elif target > len(dashboard_bodies):
                dashboard_in_solar_mode = False
                while len(dashboard_bodies) < target:
                    if load_solar and len(dashboard_bodies) < len(solar_base):
                        dashboard_bodies.append(solar_base[len(dashboard_bodies)].copy())
                    else:
                        dashboard_bodies.append(
                            add_random_body_separate(
                                existing=dashboard_bodies,
                                rng=dashboard_rng,
                                orbit_limit_x=args.orbit_limit_x,
                                orbit_limit_y=args.orbit_limit_y,
                                orbit_limit_z=args.orbit_limit_z,
                                start_padding=args.start_padding,
                            )
                        )

            args.body_count = len(dashboard_bodies)
            return create_live_simulation(
                args,
                None,
                initial_override=dashboard_bodies,
                apply_velocity_multiplier=not dashboard_in_solar_mode,
                enforce_start_clamp=not dashboard_in_solar_mode,
                enforce_bound=not dashboard_in_solar_mode,
            )

        Renderer().plot_live(
            simulation=live_sim,
            title=title,
            frame_stride=effective_stride,
            frame_skip=max(1, int(args.frame_skip)),
            interval_ms=effective_interval,
            escape_radius=args.escape_radius,
            orbit_limit_x=args.orbit_limit_x,
            orbit_limit_y=args.orbit_limit_y,
            orbit_limit_z=args.orbit_limit_z,
            refresh_fn=refresh_fn,
        )


if __name__ == "__main__":
    main()
