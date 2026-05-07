from __future__ import annotations

from typing import Callable
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import numpy as np


COLLISION_DISTANCE_SCALE = 0.2


class Renderer:
    def __init__(self) -> None:
        plt.style.use("seaborn-v0_8-darkgrid")
        self._refresh_button = None
        self._refresh_button_ax = None
        self._stop_button = None
        self._stop_button_ax = None
        self._play_button = None
        self._play_button_ax = None
        self._speed_slider = None
        self._speed_slider_ax = None
        self._count_slider = None
        self._count_slider_ax = None
        self._solar_button = None
        self._solar_button_ax = None

    def plot_results(
        self,
        trajectories: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray | None,
        body_names: list[str] | None,
        time: np.ndarray,
        energy: np.ndarray,
        divergence: np.ndarray,
        perturb_mag: np.ndarray,
        branch_divergence: np.ndarray | None = None,
        title: str = "Three-Body Simulation",
        animate: bool = True,
        frame_stride: int = 4,
        frame_skip: int = 2,
        trail_length: int = 720,
        interval_ms: int = 25,
        escape_radius: float = 6.0,
        orbit_limit_x: float = 12.0,
        orbit_limit_y: float = 8.0,
        orbit_limit_z: float = 8.0,
        refresh_fn: Callable | None = None,
    ) -> None:
        fig = plt.figure(figsize=(14, 10))
        ax_orbit = fig.add_subplot(111, projection="3d")
        self._animation = None

        colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#8D99AE", "#264653", "#E9C46A", "#118AB2"]

        data = {
            "traj": trajectories,
            "vel": velocities,
            "masses": masses if masses is not None else np.ones(trajectories.shape[1], dtype=float),
            "time": time,
            "energy": energy,
            "divergence": divergence,
            "perturb": perturb_mag,
            "branch_div": branch_divergence,
            "frame_pos": 0.0,
            "restart": False,
            "playback_scale": 1.0,
            "escaped_lock": False,
            "suppress_count_event": False,
            "is_paused": False,
            "finished": False,
            "body_names": body_names if body_names is not None else [f"Body {i + 1}" for i in range(trajectories.shape[1])],
        }

        orbit_lines: list = []
        body_surfaces: list = []
        dot_texts: list = []
        status_texts: list = []

        sphere_u = np.linspace(0.0, 2.0 * np.pi, 16)
        sphere_v = np.linspace(0.0, np.pi, 12)
        body_radii = np.ones(data["traj"].shape[1], dtype=float) * 0.25

        restart_text = ax_orbit.text2D(
            0.5, 0.97, "", transform=ax_orbit.transAxes,
            ha="center", va="top", fontsize=9, color="#E63946",
        )
        indicator_text = ax_orbit.text2D(
            0.02,
            0.02,
            "",
            transform=ax_orbit.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
        )

        def _n_bodies() -> int:
            return int(data["traj"].shape[1])

        def _sphere_mesh(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = center[0] + radius * np.outer(np.cos(sphere_u), np.sin(sphere_v))
            y = center[1] + radius * np.outer(np.sin(sphere_u), np.sin(sphere_v))
            z = center[2] + radius * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
            return x, y, z

        def _recompute_body_radii() -> np.ndarray:
            m = np.array(data["masses"], dtype=float)
            mmean = float(np.mean(m)) + 1e-12
            n = _n_bodies()
            base = max(0.05, 0.16 - 0.010 * max(0, n - 3))
            return base * (0.85 + 0.30 * np.power(m / mmean, 1.0 / 9.0))

        def _rebuild_indicator_artists() -> None:
            for t in dot_texts:
                t.remove()
            for t in status_texts:
                t.remove()
            dot_texts.clear()
            status_texts.clear()

            n = _n_bodies()
            y_top = 0.14
            spacing = 0.022
            for i in range(n):
                y = y_top - i * spacing
                dot_texts.append(fig.text(0.060, y, "●", color=colors[i % len(colors)], fontsize=11, ha="left", va="center"))
                status_texts.append(
                    fig.text(
                        0.072,
                        y,
                        "",
                        ha="left",
                        va="center",
                        fontsize=8,
                        family="monospace",
                        zorder=9,
                    )
                )
            indicator_text.set_text("\n" * max(1, n))

        def _rebuild_body_artists(frame_idx: int) -> None:
            nonlocal body_radii
            for ln in orbit_lines:
                ln.remove()
            for sf in body_surfaces:
                sf.remove()
            orbit_lines.clear()
            body_surfaces.clear()

            body_radii = _recompute_body_radii()
            labels = [str(data["body_names"][i]) for i in range(_n_bodies())]
            for i in range(_n_bodies()):
                line, = ax_orbit.plot([], [], [], color=colors[i % len(colors)], lw=1.4, alpha=0.7)
                c0 = data["traj"][frame_idx, i]
                sx, sy, sz = _sphere_mesh(c0, float(body_radii[i]))
                surf = ax_orbit.plot_surface(
                    sx,
                    sy,
                    sz,
                    color=colors[i % len(colors)],
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                    alpha=0.95,
                )
                orbit_lines.append(line)
                body_surfaces.append(surf)

            legend_handles = [
                Line2D([0], [0], color=colors[i % len(colors)], lw=2.0, marker="o", markersize=8, label=labels[i])
                for i in range(_n_bodies())
            ]
            ax_orbit.legend(handles=legend_handles, loc="upper right")

        def _refresh_spheres(frame_idx: int) -> None:
            for i in range(_n_bodies()):
                body_surfaces[i].remove()
                center = data["traj"][frame_idx, i]
                sx, sy, sz = _sphere_mesh(center, float(body_radii[i]))
                body_surfaces[i] = ax_orbit.plot_surface(
                    sx,
                    sy,
                    sz,
                    color=colors[i % len(colors)],
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                    alpha=0.95,
                )

        def _frame_center(frame_idx: int) -> np.ndarray:
            return np.mean(data["traj"][frame_idx], axis=0)

        def _reset_axes_limits(center: np.ndarray) -> None:
            lim_x = max(2.0, float(orbit_limit_x))
            lim_y = max(2.0, float(orbit_limit_y))
            lim_z = max(2.0, float(orbit_limit_z))
            current_elev = float(getattr(ax_orbit, "elev", 28.0))
            current_azim = float(getattr(ax_orbit, "azim", 45.0))
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            ax_orbit.set_xlim(cx - lim_x, cx + lim_x)
            ax_orbit.set_ylim(cy - lim_y, cy + lim_y)
            ax_orbit.set_zlim(cz - lim_z, cz + lim_z)
            ax_orbit.set_xticks(np.linspace(cx - lim_x, cx + lim_x, 13))
            ax_orbit.set_yticks(np.linspace(cy - lim_y, cy + lim_y, 9))
            ax_orbit.set_zticks(np.linspace(cz - lim_z, cz + lim_z, 9))
            ax_orbit.set_box_aspect((2 * lim_x, 2 * lim_y, 2 * lim_z))
            ax_orbit.view_init(elev=current_elev, azim=current_azim)
            ax_orbit.grid(True, alpha=0.35)

        def _indicator_lines(frame_idx: int) -> list[str]:
            vel = data["vel"][frame_idx]
            body_masses = data["masses"]
            lines: list[str] = []
            for i in range(_n_bodies()):
                vx, vy, vz = float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])
                speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
                azimuth = float(np.degrees(np.arctan2(vy, vx)))
                elevation = float(np.degrees(np.arctan2(vz, np.hypot(vx, vy))))
                lines.append(f"m={body_masses[i]:.2f}  v={speed:.3f}  az={azimuth:+6.1f}  el={elevation:+5.1f} deg")
            return lines

        def _request_new_run(
            requested_body_count: int | None = None,
            load_solar: bool = False,
            randomize_existing: bool = False,
        ):
            if refresh_fn is None:
                return None
            try:
                return refresh_fn(
                    requested_body_count=requested_body_count,
                    load_solar=load_solar,
                    randomize_existing=randomize_existing,
                )
            except TypeError:
                return refresh_fn()

        ax_orbit.set_title("Orbit Paths (3D)")
        ax_orbit.set_xlabel("x")
        ax_orbit.set_ylabel("y")
        ax_orbit.set_zlabel("z")

        ax_orbit.view_init(elev=28, azim=45)
        _rebuild_body_artists(0)
        _rebuild_indicator_artists()
        _reset_axes_limits(_frame_center(0))

        panel_ax = fig.add_axes([0.02, 0.52, 0.20, 0.20])
        panel_ax.set_facecolor("#f1f3f5")
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
        panel_ax.set_frame_on(True)
        panel_ax.text(0.05, 0.88, "Dashboard", fontsize=10, fontweight="bold", transform=panel_ax.transAxes)

        count_ax = fig.add_axes([0.04, 0.62, 0.16, 0.03])
        count_slider = Slider(count_ax, "Bodies", 1, 8, valinit=max(1, _n_bodies()), valstep=1)
        solar_ax = fig.add_axes([0.04, 0.56, 0.16, 0.045])
        solar_button = Button(solar_ax, "Earth / Solar")

        refresh_ax = fig.add_axes([0.82, 0.02, 0.14, 0.05])
        refresh_button = Button(refresh_ax, "Refresh")
        stop_ax = fig.add_axes([0.66, 0.02, 0.14, 0.05])
        stop_button = Button(stop_ax, "Stop")
        play_ax = fig.add_axes([0.50, 0.02, 0.14, 0.05])
        play_button = Button(play_ax, "Play")
        speed_ax = fig.add_axes([0.24, 0.03, 0.20, 0.03])
        speed_slider = Slider(speed_ax, "Time Flow", 0.25, 4.0, valinit=1.0, valstep=0.05)

        self._count_slider = count_slider
        self._count_slider_ax = count_ax
        self._solar_button = solar_button
        self._solar_button_ax = solar_ax
        self._refresh_button = refresh_button
        self._refresh_button_ax = refresh_ax
        self._stop_button = stop_button
        self._stop_button_ax = stop_ax
        self._play_button = play_button
        self._play_button_ax = play_ax
        self._speed_slider = speed_slider
        self._speed_slider_ax = speed_ax

        def _escaped_body_index(frame_idx: int) -> int | None:
            traj = data["traj"]
            positions = traj[frame_idx]
            center = np.mean(positions, axis=0)
            rel = positions - center
            lim_x = max(2.0, float(orbit_limit_x))
            lim_y = max(2.0, float(orbit_limit_y))
            lim_z = max(2.0, float(orbit_limit_z))
            escaped_mask = (
                (np.abs(rel[:, 0]) > lim_x)
                | (np.abs(rel[:, 1]) > lim_y)
                | (np.abs(rel[:, 2]) > lim_z)
            )
            escaped_indices = np.where(escaped_mask)[0]
            if escaped_indices.size > 0:
                return int(escaped_indices[0])

            if escape_radius > 0.0:
                rel_norms = np.linalg.norm(rel, axis=1)
                escaped_radius = np.where(rel_norms > escape_radius)[0]
                if escaped_radius.size > 0:
                    return int(escaped_radius[0])
            return None

        def _collided_body_pair(frame_idx: int) -> tuple[int, int] | None:
            positions = data["traj"][frame_idx]
            for i in range(_n_bodies()):
                for j in range(i + 1, _n_bodies()):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist <= float((body_radii[i] + body_radii[j]) * COLLISION_DISTANCE_SCALE):
                        return i, j
            return None

        base_stride = max(1, int(frame_stride)) * max(1, int(frame_skip))

        def _load_data(new_run) -> None:
            data["traj"] = new_run.trajectories
            data["vel"] = new_run.velocities
            data["masses"] = new_run.masses
            data["body_names"] = list(new_run.body_names)
            data["time"] = new_run.time
            data["energy"] = new_run.energy
            data["divergence"] = new_run.divergence
            data["perturb"] = new_run.perturbation_mag
            data["branch_div"] = new_run.branch_divergence
            data["frame_pos"] = 0.0
            data["restart"] = False
            data["escaped_lock"] = False
            data["is_paused"] = False
            data["finished"] = False

            _rebuild_body_artists(0)
            _rebuild_indicator_artists()
            _reset_axes_limits(_frame_center(0))
            for line in orbit_lines:
                line.set_data_3d([], [], [])
            _refresh_spheres(0)

        def _on_refresh(_event) -> None:
            restart_text.set_text("Refreshing…")
            data["escaped_lock"] = False
            data["is_paused"] = False
            target = int(round(float(count_slider.val)))
            new_run = _request_new_run(requested_body_count=target, load_solar=False, randomize_existing=True)
            if new_run is not None:
                _load_data(new_run)
            restart_text.set_text("")
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.start()
            fig.canvas.draw_idle()

        def _on_stop(_event) -> None:
            data["is_paused"] = True
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.stop()
            restart_text.set_text("Paused")
            fig.canvas.draw_idle()

        def _on_play(_event) -> None:
            if data["escaped_lock"]:
                restart_text.set_text("Refresh required")
                fig.canvas.draw_idle()
                return
            if data["finished"]:
                restart_text.set_text("Refresh required")
                fig.canvas.draw_idle()
                return
            data["is_paused"] = False
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.start()
                if restart_text.get_text() == "Paused":
                    restart_text.set_text("")
                fig.canvas.draw_idle()

        def _on_speed_change(value: float) -> None:
            data["playback_scale"] = float(value)

        def _on_body_count_change(value: float) -> None:
            if data["suppress_count_event"]:
                return
            target = int(round(value))
            new_run = _request_new_run(requested_body_count=target, load_solar=False)
            if new_run is not None:
                _load_data(new_run)
                restart_text.set_text("")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.start()
                fig.canvas.draw_idle()

        def _on_solar(_event) -> None:
            data["suppress_count_event"] = True
            count_slider.set_val(8)
            data["suppress_count_event"] = False
            new_run = _request_new_run(requested_body_count=8, load_solar=True)
            if new_run is not None:
                _load_data(new_run)
                restart_text.set_text("Solar preset loaded")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.start()
                fig.canvas.draw_idle()

        refresh_button.on_clicked(_on_refresh)
        stop_button.on_clicked(_on_stop)
        play_button.on_clicked(_on_play)
        speed_slider.on_changed(_on_speed_change)
        count_slider.on_changed(_on_body_count_change)
        solar_button.on_clicked(_on_solar)

        if not animate:
            traj = data["traj"]
            for i in range(_n_bodies()):
                orbit_lines[i].set_data_3d(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2])
            _refresh_spheres(traj.shape[0] - 1)
            lines = _indicator_lines(traj.shape[0] - 1)
            for i in range(_n_bodies()):
                status_texts[i].set_text(lines[i])
            _reset_axes_limits(_frame_center(traj.shape[0] - 1))
            fig.suptitle(title)
            fig.tight_layout(rect=[0.24, 0.08, 1, 1])
            plt.show()
            return

        def _artists() -> list:
            return [*orbit_lines, *body_surfaces, indicator_text, restart_text, *dot_texts, *status_texts]

        def update(_tick: int) -> list:
            if data["restart"] and refresh_fn is not None:
                restart_text.set_text("Restarting…")
                fig.canvas.draw_idle()
                new_run = _request_new_run(requested_body_count=int(round(float(count_slider.val))), load_solar=False)
                if new_run is not None:
                    _load_data(new_run)
                restart_text.set_text("")

            traj = data["traj"]
            fi = int(data["frame_pos"])
            n_frames = traj.shape[0]

            if fi >= n_frames:
                data["finished"] = True
                data["is_paused"] = True
                restart_text.set_text("Simulation complete")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            if data["is_paused"]:
                return _artists()

            if data["escaped_lock"]:
                return _artists()

            escaped_body = _escaped_body_index(fi) if fi > 0 else None
            if escaped_body is not None:
                data["escaped_lock"] = True
                restart_text.set_text(f"body {escaped_body + 1} left the gravitation field")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            collided_pair = _collided_body_pair(fi) if fi > 0 else None
            if collided_pair is not None:
                first, second = collided_pair
                data["is_paused"] = True
                data["finished"] = True
                restart_text.set_text(
                    f"{data['body_names'][first]} collided with {data['body_names'][second]}"
                )
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            for i in range(_n_bodies()):
                start_idx = max(0, fi + 1 - max(1, int(trail_length)))
                orbit_lines[i].set_data_3d(
                    traj[start_idx: fi + 1, i, 0],
                    traj[start_idx: fi + 1, i, 1],
                    traj[start_idx: fi + 1, i, 2],
                )
            _refresh_spheres(fi)
            _reset_axes_limits(_frame_center(fi))

            lines = _indicator_lines(fi)
            for i in range(_n_bodies()):
                status_texts[i].set_text(lines[i])

            adaptive_stride = base_stride + max(0, _n_bodies() - 3)
            data["frame_pos"] = data["frame_pos"] + adaptive_stride * float(data["playback_scale"])
            return _artists()

        self._animation = FuncAnimation(
            fig,
            update,
            interval=max(1, int(interval_ms)),
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )

        fig.suptitle(title)
        fig.tight_layout(rect=[0.24, 0.08, 1, 1])
        plt.show()

    def plot_live(
        self,
        simulation,
        title: str = "Three-Body Simulation",
        frame_stride: int = 4,
        frame_skip: int = 2,
        trail_length: int = 720,
        interval_ms: int = 25,
        escape_radius: float = 6.0,
        orbit_limit_x: float = 12.0,
        orbit_limit_y: float = 8.0,
        orbit_limit_z: float = 8.0,
        refresh_fn: Callable | None = None,
    ) -> None:
        fig = plt.figure(figsize=(14, 10))
        ax_orbit = fig.add_subplot(111, projection="3d")
        self._animation = None

        colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#8D99AE", "#264653", "#E9C46A", "#118AB2"]
        initial = simulation.snapshot()

        data = {
            "sim": simulation,
            "snapshot": initial,
            "playback_scale": 1.0,
            "escaped_lock": False,
            "suppress_count_event": False,
            "is_paused": False,
            "finished": False,
            "step_accumulator": 0.0,
            "history": [[initial.positions[i].copy()] for i in range(initial.positions.shape[0])],
        }

        orbit_lines: list = []
        body_surfaces: list = []
        dot_texts: list = []
        status_texts: list = []

        sphere_u = np.linspace(0.0, 2.0 * np.pi, 16)
        sphere_v = np.linspace(0.0, np.pi, 12)
        body_radii = np.ones(initial.positions.shape[0], dtype=float) * 0.25

        restart_text = ax_orbit.text2D(
            0.5, 0.97, "", transform=ax_orbit.transAxes,
            ha="center", va="top", fontsize=9, color="#E63946",
        )
        indicator_text = ax_orbit.text2D(
            0.02,
            0.02,
            "",
            transform=ax_orbit.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666666", "boxstyle": "round,pad=0.3"},
        )

        def _snapshot():
            return data["snapshot"]

        def _n_bodies() -> int:
            return int(_snapshot().positions.shape[0])

        def _sphere_mesh(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = center[0] + radius * np.outer(np.cos(sphere_u), np.sin(sphere_v))
            y = center[1] + radius * np.outer(np.sin(sphere_u), np.sin(sphere_v))
            z = center[2] + radius * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
            return x, y, z

        def _recompute_body_radii() -> np.ndarray:
            m = np.array(_snapshot().masses, dtype=float)
            mmean = float(np.mean(m)) + 1e-12
            n = _n_bodies()
            base = max(0.05, 0.16 - 0.010 * max(0, n - 3))
            return base * (0.85 + 0.30 * np.power(m / mmean, 1.0 / 9.0))

        def _rebuild_indicator_artists() -> None:
            for t in dot_texts:
                t.remove()
            for t in status_texts:
                t.remove()
            dot_texts.clear()
            status_texts.clear()

            n = _n_bodies()
            y_top = 0.14
            spacing = 0.022
            for i in range(n):
                y = y_top - i * spacing
                dot_texts.append(fig.text(0.060, y, "●", color=colors[i % len(colors)], fontsize=11, ha="left", va="center"))
                status_texts.append(
                    fig.text(0.072, y, "", ha="left", va="center", fontsize=8, family="monospace", zorder=9)
                )
            indicator_text.set_text("\n" * max(1, n))

        def _rebuild_body_artists() -> None:
            nonlocal body_radii
            for ln in orbit_lines:
                ln.remove()
            for sf in body_surfaces:
                sf.remove()
            orbit_lines.clear()
            body_surfaces.clear()

            body_radii = _recompute_body_radii()
            labels = [str(_snapshot().body_names[i]) for i in range(_n_bodies())]
            for i in range(_n_bodies()):
                line, = ax_orbit.plot([], [], [], color=colors[i % len(colors)], lw=1.4, alpha=0.7)
                c0 = _snapshot().positions[i]
                sx, sy, sz = _sphere_mesh(c0, float(body_radii[i]))
                surf = ax_orbit.plot_surface(
                    sx, sy, sz, color=colors[i % len(colors)], linewidth=0, antialiased=True, shade=True, alpha=0.95
                )
                orbit_lines.append(line)
                body_surfaces.append(surf)

            legend_handles = [
                Line2D([0], [0], color=colors[i % len(colors)], lw=2.0, marker="o", markersize=8, label=labels[i])
                for i in range(_n_bodies())
            ]
            ax_orbit.legend(handles=legend_handles, loc="upper right")

        def _refresh_spheres() -> None:
            for i in range(_n_bodies()):
                body_surfaces[i].remove()
                center = _snapshot().positions[i]
                sx, sy, sz = _sphere_mesh(center, float(body_radii[i]))
                body_surfaces[i] = ax_orbit.plot_surface(
                    sx, sy, sz, color=colors[i % len(colors)], linewidth=0, antialiased=True, shade=True, alpha=0.95
                )

        def _frame_center() -> np.ndarray:
            return np.mean(_snapshot().positions, axis=0)

        def _reset_axes_limits(center: np.ndarray) -> None:
            lim_x = max(2.0, float(orbit_limit_x))
            lim_y = max(2.0, float(orbit_limit_y))
            lim_z = max(2.0, float(orbit_limit_z))
            current_elev = float(getattr(ax_orbit, "elev", 28.0))
            current_azim = float(getattr(ax_orbit, "azim", 45.0))
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            ax_orbit.set_xlim(cx - lim_x, cx + lim_x)
            ax_orbit.set_ylim(cy - lim_y, cy + lim_y)
            ax_orbit.set_zlim(cz - lim_z, cz + lim_z)
            ax_orbit.set_xticks(np.linspace(cx - lim_x, cx + lim_x, 13))
            ax_orbit.set_yticks(np.linspace(cy - lim_y, cy + lim_y, 9))
            ax_orbit.set_zticks(np.linspace(cz - lim_z, cz + lim_z, 9))
            ax_orbit.set_box_aspect((2 * lim_x, 2 * lim_y, 2 * lim_z))
            ax_orbit.view_init(elev=current_elev, azim=current_azim)
            ax_orbit.grid(True, alpha=0.35)

        def _indicator_lines() -> list[str]:
            vel = _snapshot().velocities
            body_masses = _snapshot().masses
            lines: list[str] = []
            for i in range(_n_bodies()):
                vx, vy, vz = float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])
                speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
                azimuth = float(np.degrees(np.arctan2(vy, vx)))
                elevation = float(np.degrees(np.arctan2(vz, np.hypot(vx, vy))))
                lines.append(f"m={body_masses[i]:.2f}  v={speed:.3f}  az={azimuth:+6.1f}  el={elevation:+5.1f} deg")
            return lines

        def _request_new_run(requested_body_count: int | None = None, load_solar: bool = False, randomize_existing: bool = False):
            if refresh_fn is None:
                return None
            return refresh_fn(
                requested_body_count=requested_body_count,
                load_solar=load_solar,
                randomize_existing=randomize_existing,
            )

        def _escaped_body_index() -> int | None:
            positions = _snapshot().positions
            center = np.mean(positions, axis=0)
            rel = positions - center
            lim_x = max(2.0, float(orbit_limit_x))
            lim_y = max(2.0, float(orbit_limit_y))
            lim_z = max(2.0, float(orbit_limit_z))
            escaped_mask = (
                (np.abs(rel[:, 0]) > lim_x)
                | (np.abs(rel[:, 1]) > lim_y)
                | (np.abs(rel[:, 2]) > lim_z)
            )
            escaped_indices = np.where(escaped_mask)[0]
            if escaped_indices.size > 0:
                return int(escaped_indices[0])
            if escape_radius > 0.0:
                rel_norms = np.linalg.norm(rel, axis=1)
                escaped_radius = np.where(rel_norms > escape_radius)[0]
                if escaped_radius.size > 0:
                    return int(escaped_radius[0])
            return None

        def _collided_body_pair() -> tuple[int, int] | None:
            positions = _snapshot().positions
            for i in range(_n_bodies()):
                for j in range(i + 1, _n_bodies()):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist <= float((body_radii[i] + body_radii[j]) * COLLISION_DISTANCE_SCALE):
                        return i, j
            return None

        def _load_sim(new_sim) -> None:
            data["sim"] = new_sim
            data["snapshot"] = new_sim.snapshot()
            data["escaped_lock"] = False
            data["is_paused"] = False
            data["finished"] = False
            data["step_accumulator"] = 0.0
            data["history"] = [[data["snapshot"].positions[i].copy()] for i in range(data["snapshot"].positions.shape[0])]
            _rebuild_body_artists()
            _rebuild_indicator_artists()
            _refresh_spheres()
            _reset_axes_limits(_frame_center())
            for line in orbit_lines:
                line.set_data_3d([], [], [])

        ax_orbit.set_title("Orbit Paths (3D)")
        ax_orbit.set_xlabel("x")
        ax_orbit.set_ylabel("y")
        ax_orbit.set_zlabel("z")
        ax_orbit.view_init(elev=28, azim=45)
        _rebuild_body_artists()
        _rebuild_indicator_artists()
        _reset_axes_limits(_frame_center())

        panel_ax = fig.add_axes([0.02, 0.52, 0.20, 0.20])
        panel_ax.set_facecolor("#f1f3f5")
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
        panel_ax.set_frame_on(True)
        panel_ax.text(0.05, 0.88, "Dashboard", fontsize=10, fontweight="bold", transform=panel_ax.transAxes)

        count_ax = fig.add_axes([0.04, 0.62, 0.16, 0.03])
        count_slider = Slider(count_ax, "Bodies", 1, 8, valinit=max(1, _n_bodies()), valstep=1)
        solar_ax = fig.add_axes([0.04, 0.56, 0.16, 0.045])
        solar_button = Button(solar_ax, "Earth / Solar")
        refresh_ax = fig.add_axes([0.82, 0.02, 0.14, 0.05])
        refresh_button = Button(refresh_ax, "Refresh")
        stop_ax = fig.add_axes([0.66, 0.02, 0.14, 0.05])
        stop_button = Button(stop_ax, "Stop")
        play_ax = fig.add_axes([0.50, 0.02, 0.14, 0.05])
        play_button = Button(play_ax, "Play")
        speed_ax = fig.add_axes([0.24, 0.03, 0.20, 0.03])
        speed_slider = Slider(speed_ax, "Time Flow", 0.25, 4.0, valinit=1.0, valstep=0.05)

        self._count_slider = count_slider
        self._count_slider_ax = count_ax
        self._solar_button = solar_button
        self._solar_button_ax = solar_ax
        self._refresh_button = refresh_button
        self._refresh_button_ax = refresh_ax
        self._stop_button = stop_button
        self._stop_button_ax = stop_ax
        self._play_button = play_button
        self._play_button_ax = play_ax
        self._speed_slider = speed_slider
        self._speed_slider_ax = speed_ax

        base_stride = max(1, int(frame_stride)) * max(1, int(frame_skip))

        def _on_refresh(_event) -> None:
            restart_text.set_text("Refreshing…")
            target = int(round(float(count_slider.val)))
            new_sim = _request_new_run(requested_body_count=target, load_solar=False, randomize_existing=True)
            if new_sim is not None:
                _load_sim(new_sim)
            restart_text.set_text("")
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.start()
            fig.canvas.draw_idle()

        def _on_stop(_event) -> None:
            data["is_paused"] = True
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.stop()
            restart_text.set_text("Paused")
            fig.canvas.draw_idle()

        def _on_play(_event) -> None:
            if data["escaped_lock"] or data["finished"]:
                restart_text.set_text("Refresh required")
                fig.canvas.draw_idle()
                return
            data["is_paused"] = False
            if self._animation is not None and self._animation.event_source is not None:
                self._animation.event_source.start()
            if restart_text.get_text() == "Paused":
                restart_text.set_text("")
            fig.canvas.draw_idle()

        def _on_speed_change(value: float) -> None:
            data["playback_scale"] = float(value)

        def _on_body_count_change(value: float) -> None:
            if data["suppress_count_event"]:
                return
            target = int(round(value))
            new_sim = _request_new_run(requested_body_count=target, load_solar=False)
            if new_sim is not None:
                _load_sim(new_sim)
                restart_text.set_text("")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.start()
                fig.canvas.draw_idle()

        def _on_solar(_event) -> None:
            data["suppress_count_event"] = True
            count_slider.set_val(8)
            data["suppress_count_event"] = False
            new_sim = _request_new_run(requested_body_count=8, load_solar=True)
            if new_sim is not None:
                _load_sim(new_sim)
                restart_text.set_text("Solar preset loaded")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.start()
                fig.canvas.draw_idle()

        refresh_button.on_clicked(_on_refresh)
        stop_button.on_clicked(_on_stop)
        play_button.on_clicked(_on_play)
        speed_slider.on_changed(_on_speed_change)
        count_slider.on_changed(_on_body_count_change)
        solar_button.on_clicked(_on_solar)

        def _artists() -> list:
            return [*orbit_lines, *body_surfaces, indicator_text, restart_text, *dot_texts, *status_texts]

        lines = _indicator_lines()
        for i in range(_n_bodies()):
            status_texts[i].set_text(lines[i])

        def update(_tick: int) -> list:
            if data["is_paused"] or data["escaped_lock"] or data["finished"]:
                return _artists()

            adaptive_stride = base_stride + max(0, _n_bodies() - 3)
            data["step_accumulator"] += adaptive_stride * float(data["playback_scale"])
            steps_to_take = int(data["step_accumulator"])
            if steps_to_take <= 0:
                return _artists()
            data["step_accumulator"] -= steps_to_take

            data["snapshot"] = data["sim"].step(steps_to_take)
            snapshot = _snapshot()

            if snapshot.finished:
                data["finished"] = True
                data["is_paused"] = True
                restart_text.set_text("Simulation complete")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            escaped_body = _escaped_body_index()
            if escaped_body is not None:
                data["escaped_lock"] = True
                restart_text.set_text(f"body {escaped_body + 1} left the gravitation field")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            collided_pair = _collided_body_pair()
            if collided_pair is not None:
                first, second = collided_pair
                data["finished"] = True
                data["is_paused"] = True
                restart_text.set_text(f"{snapshot.body_names[first]} collided with {snapshot.body_names[second]}")
                if self._animation is not None and self._animation.event_source is not None:
                    self._animation.event_source.stop()
                return _artists()

            for i in range(_n_bodies()):
                data["history"][i].append(snapshot.positions[i].copy())
                if len(data["history"][i]) > max(1, int(trail_length)):
                    data["history"][i] = data["history"][i][-int(trail_length):]
                hist = np.stack(data["history"][i], axis=0)
                orbit_lines[i].set_data_3d(hist[:, 0], hist[:, 1], hist[:, 2])

            _refresh_spheres()
            _reset_axes_limits(_frame_center())
            lines = _indicator_lines()
            for i in range(_n_bodies()):
                status_texts[i].set_text(lines[i])
            return _artists()

        self._animation = FuncAnimation(
            fig,
            update,
            interval=max(1, int(interval_ms)),
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )

        fig.suptitle(title)
        fig.tight_layout(rect=[0.24, 0.08, 1, 1])
        plt.show()
