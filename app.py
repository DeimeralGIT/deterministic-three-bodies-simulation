from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from classes import DeterminismAnalyzer
from simulation import (
    DEFAULT_ORBIT_LIMIT_X,
    DEFAULT_ORBIT_LIMIT_Y,
    DEFAULT_ORBIT_LIMIT_Z,
    DEFAULT_START_PADDING,
    MAX_BODIES,
    random_initial_conditions,
    run_simulation,
    solar_system_initial_conditions,
)


COLORS = ["#e63946", "#1d3557", "#2a9d8f", "#f4a261", "#8d99ae", "#264653", "#e9c46a", "#118ab2"]


def configured_password() -> str:
    try:
        secret_value = st.secrets.get("APP_PASSWORD")
    except Exception:
        secret_value = None
    return str(secret_value or os.environ.get("APP_PASSWORD") or "")


def require_password() -> bool:
    password = configured_password()
    if not password:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Three-Body Simulation")
    attempt = st.text_input("Password", type="password")
    if st.button("Enter"):
        if attempt == password:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password")
    return False


def default_args(**overrides) -> Namespace:
    values = {
        "mode": "A",
        "dt": 0.003,
        "steps": 3000,
        "integrator": "rk4",
        "noise_scale": 2e-5,
        "branches": 6,
        "seed": 7,
        "gravity": 1.0,
        "softening": 1e-5,
        "speed": 1.0,
        "bound": False,
        "random": True,
        "escape_radius": 0.0,
        "adaptive": False,
        "no_show": True,
        "static": False,
        "frame_stride": 4,
        "frame_skip": 2,
        "interval_ms": 25,
        "body_count": 3,
        "orbit_limit_x": DEFAULT_ORBIT_LIMIT_X,
        "orbit_limit_y": DEFAULT_ORBIT_LIMIT_Y,
        "orbit_limit_z": DEFAULT_ORBIT_LIMIT_Z,
        "start_padding": DEFAULT_START_PADDING,
        "masses": None,
        "positions": None,
        "velocities": None,
    }
    values.update(overrides)
    return Namespace(**values)


@st.cache_data(show_spinner=False)
def cached_run(config: tuple) -> tuple[object, object, str]:
    (
        mode,
        dt,
        steps,
        integrator,
        noise_scale,
        branches,
        seed,
        gravity,
        softening,
        adaptive,
        body_count,
        orbit_limit_x,
        orbit_limit_y,
        orbit_limit_z,
        start_padding,
        preset,
    ) = config

    args = default_args(
        mode=mode,
        dt=dt,
        steps=steps,
        integrator=integrator,
        noise_scale=noise_scale,
        branches=branches,
        seed=seed,
        gravity=gravity,
        softening=softening,
        adaptive=adaptive,
        body_count=body_count,
        orbit_limit_x=orbit_limit_x,
        orbit_limit_y=orbit_limit_y,
        orbit_limit_z=orbit_limit_z,
        start_padding=start_padding,
    )

    if preset == "Solar":
        initial = solar_system_initial_conditions(gravity, orbit_limit_x, orbit_limit_y, orbit_limit_z, start_padding)
        initial = initial[:body_count]
        run = run_simulation(
            args,
            initial_override=initial,
            apply_velocity_multiplier=False,
            enforce_start_clamp=False,
            enforce_bound=False,
        )
    else:
        rng = np.random.default_rng(seed)
        initial = random_initial_conditions(
            rng,
            orbit_limit_x=orbit_limit_x,
            orbit_limit_y=orbit_limit_y,
            orbit_limit_z=orbit_limit_z,
            start_padding=start_padding,
            body_count=body_count,
        )
        run = run_simulation(args, initial_override=initial)

    analyzer = DeterminismAnalyzer()
    summary = analyzer.summarize(
        mode=mode,
        divergence=run.divergence,
        energy=run.energy,
        momentum_mag=run.momentum_mag,
        perturb_mag=run.perturbation_mag,
        cancellation=run.cancellation,
    )
    return run, summary, analyzer.philosophical_report(summary)


def frame_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames).astype(int))


def orbit_figure(run, orbit_limit_x: float, orbit_limit_y: float, orbit_limit_z: float, max_frames: int, trail: int) -> go.Figure:
    indices = frame_indices(run.trajectories.shape[0], max_frames)
    first = int(indices[0])
    center = np.mean(run.trajectories[first], axis=0)

    def traces_for(frame_idx: int) -> list[go.Scatter3d]:
        traces: list[go.Scatter3d] = []
        start = max(0, frame_idx - trail)
        for body_idx, name in enumerate(run.body_names):
            color = COLORS[body_idx % len(COLORS)]
            traj = run.trajectories[start : frame_idx + 1, body_idx, :]
            pos = run.trajectories[frame_idx, body_idx, :]
            traces.append(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    line={"width": 4, "color": color},
                    name=f"{name} trail",
                    showlegend=False,
                )
            )
            traces.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers+text",
                    marker={
                        "size": max(5, 9 * float(run.masses[body_idx] / np.mean(run.masses)) ** (1.0 / 9.0)),
                        "color": color,
                    },
                    text=[name],
                    textposition="top center",
                    name=name,
                    showlegend=True,
                )
            )
        return traces

    frames = [
        go.Frame(
            data=traces_for(int(idx)),
            name=str(int(idx)),
            layout=go.Layout(title_text=f"t = {run.time[int(idx)]:.3f}"),
        )
        for idx in indices
    ]

    fig = go.Figure(data=traces_for(first), frames=frames)
    fig.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        title=f"t = {run.time[first]:.3f}",
        scene={
            "xaxis": {"title": "x", "range": [center[0] - orbit_limit_x, center[0] + orbit_limit_x]},
            "yaxis": {"title": "y", "range": [center[1] - orbit_limit_y, center[1] + orbit_limit_y]},
            "zaxis": {"title": "z", "range": [center[2] - orbit_limit_z, center[2] + orbit_limit_z]},
            "aspectmode": "manual",
            "aspectratio": {"x": orbit_limit_x, "y": orbit_limit_y, "z": orbit_limit_z},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 35, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.18,
                "y": 0.02,
                "len": 0.76,
                "currentvalue": {"prefix": "step "},
                "steps": [
                    {
                        "label": str(int(idx)),
                        "method": "animate",
                        "args": [
                            [str(int(idx))],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                        ],
                    }
                    for idx in indices
                ],
            }
        ],
    )
    return fig


def metric_columns(summary) -> None:
    cols = st.columns(4)
    cols[0].metric("Max divergence", f"{summary.max_divergence:.2e}")
    cols[1].metric("Energy drift", f"{summary.relative_energy_drift:.2e}")
    cols[2].metric("Momentum drift", f"{summary.max_momentum_drift:.2e}")
    cols[3].metric("Chaos amp", f"{summary.chaos_amplification:.2e}")


def main() -> None:
    st.set_page_config(page_title="Three-Body Simulation", page_icon="3", layout="wide")
    if not require_password():
        return

    st.title("Three-Body Simulation")

    if "seed_bump" not in st.session_state:
        st.session_state.seed_bump = 0

    with st.sidebar:
        st.header("Run")
        preset = st.radio("Preset", ["Random", "Solar"], index=0, horizontal=True)
        mode = st.selectbox("Mode", ["A", "B", "C", "D"], index=0)
        integrator = st.selectbox("Integrator", ["rk4", "verlet"], index=0)
        body_count = st.slider("Bodies", 1, MAX_BODIES, 8 if preset == "Solar" else 3)
        steps = st.slider("Steps", 300, 12000, 3000, step=100)
        dt = st.number_input("dt", min_value=0.0001, max_value=0.05, value=0.003, step=0.0005, format="%.4f")
        noise_scale = st.number_input("Noise scale", min_value=0.0, max_value=0.01, value=2e-5, step=1e-5, format="%.6f")
        branches = st.slider("Branches", 2, 16, 6)
        gravity = st.slider("Gravity", 0.1, 5.0, 1.0, step=0.1)
        softening = st.number_input("Softening", min_value=0.0, max_value=0.01, value=1e-5, step=1e-5, format="%.6f")
        adaptive = st.toggle("Adaptive timestep", value=False)
        orbit_limit_x = st.slider("Orbit limit x", 2.0, 30.0, DEFAULT_ORBIT_LIMIT_X, step=0.5)
        orbit_limit_y = st.slider("Orbit limit y", 2.0, 30.0, DEFAULT_ORBIT_LIMIT_Y, step=0.5)
        orbit_limit_z = st.slider("Orbit limit z", 2.0, 30.0, DEFAULT_ORBIT_LIMIT_Z, step=0.5)
        start_padding = st.slider("Start padding", 0.1, 8.0, DEFAULT_START_PADDING, step=0.1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=7, step=1)
        if st.button("Refresh random state", use_container_width=True):
            st.session_state.seed_bump += 1
            st.cache_data.clear()

        st.header("Playback")
        max_frames = st.slider("Animation frames", 40, 500, 180, step=20)
        trail = st.slider("Trail length", 10, 1500, 500, step=10)

    effective_seed = int(seed) + int(st.session_state.seed_bump)
    config = (
        mode,
        float(dt),
        int(steps),
        integrator,
        float(noise_scale),
        int(branches),
        effective_seed,
        float(gravity),
        float(softening),
        bool(adaptive),
        int(body_count),
        float(orbit_limit_x),
        float(orbit_limit_y),
        float(orbit_limit_z),
        float(start_padding),
        preset,
    )

    with st.spinner("Running simulation..."):
        run, summary, report = cached_run(config)

    metric_columns(summary)
    st.plotly_chart(
        orbit_figure(run, float(orbit_limit_x), float(orbit_limit_y), float(orbit_limit_z), int(max_frames), int(trail)),
        use_container_width=True,
    )

    with st.expander("Analysis", expanded=True):
        st.text(report)

    with st.expander("Raw arrays"):
        st.write(
            {
                "time_points": int(run.time.shape[0]),
                "bodies": run.body_names,
                "masses": [float(m) for m in run.masses],
                "final_time": float(run.time[-1]),
            }
        )


if __name__ == "__main__":
    main()
