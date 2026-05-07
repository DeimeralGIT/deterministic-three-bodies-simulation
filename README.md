# Three-Body Simulation: Deterministic Quantum-Cancellation Hypothesis

This project simulates the classical Newtonian three-body problem and adds optional quantum-style perturbation layers to explore whether macroscopic dynamics remain effectively deterministic.

## Scientific Position

This code does **not** challenge established quantum mechanics. It is an exploratory computational framework that separates:

- verified classical mechanics,
- explicit perturbation assumptions,
- philosophical interpretation.

## Features

- Classical Newtonian three-body simulation in 2D
- Integrators:
  - RK4
  - Velocity Verlet
- Perturbation modes:
  - **A** Pure Classical
  - **B** Random Noise
  - **C** Symmetric Cancellation
  - **D** Branch Approximation
- Determinism analysis:
  - trajectory divergence
  - energy conservation
  - momentum conservation
  - perturbation cancellation ratio
  - chaos amplification estimate
- Visualization:
  - trajectory plot
  - velocity vectors
  - total energy over time
  - branch divergence (mode D)
  - perturbation magnitude over time

## Quick Start

1. Create environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run default simulation:

```bash
python simulation.py
```

Or run the browser-hostable version:

```bash
streamlit run app.py
```

3. Try a different mode:

```bash
python simulation.py --mode C --steps 8000 --dt 0.002 --integrator rk4
```

4. Branch approximation:

```bash
python simulation.py --mode D --branches 8 --steps 5000
```

## CLI Options

- `--mode` in `A|B|C|D`
- `--dt` timestep
- `--steps` number of integration steps
- `--integrator` in `rk4|verlet`
- `--noise-scale` perturbation amplitude
- `--branches` number of branches in mode D
- `--seed` random seed
- `--no-show` skip plotting windows

## Output

At completion, the program prints a philosophical/scientific analysis summary based on measured metrics.

## Working Online App

For an online working instance, deploy the Streamlit app:

- Entry point: `app.py`
- Install command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`

Render can use the included `render.yaml`. Heroku-compatible hosts can use the included `Procfile`.

This still is not a custom backend: it is one small Python web process that runs the existing simulation code and renders the controls/animation in the browser.

For a simple private gate, set an `APP_PASSWORD` environment variable on the host. If `APP_PASSWORD` is set, the app asks for that password before showing the simulation. For stronger access control, also put the service behind the host's password/auth setting, Cloudflare Access, or another invite-only gate.
