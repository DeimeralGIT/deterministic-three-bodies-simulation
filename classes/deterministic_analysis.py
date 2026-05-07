from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AnalysisSummary:
    mode: str
    max_divergence: float
    final_divergence: float
    relative_energy_drift: float
    max_momentum_drift: float
    mean_perturbation: float
    mean_cancellation: float
    chaos_amplification: float


class DeterminismAnalyzer:
    def summarize(
        self,
        mode: str,
        divergence: np.ndarray,
        energy: np.ndarray,
        momentum_mag: np.ndarray,
        perturb_mag: np.ndarray,
        cancellation: np.ndarray,
    ) -> AnalysisSummary:
        e0 = energy[0] if len(energy) else 1.0
        if abs(e0) < 1e-12:
            e0 = 1e-12
        relative_energy_drift = float(np.max(np.abs((energy - energy[0]) / e0))) if len(energy) else 0.0
        max_momentum_drift = float(np.max(np.abs(momentum_mag - momentum_mag[0]))) if len(momentum_mag) else 0.0
        max_div = float(np.max(divergence)) if len(divergence) else 0.0
        final_div = float(divergence[-1]) if len(divergence) else 0.0
        mean_pert = float(np.mean(perturb_mag)) if len(perturb_mag) else 0.0
        mean_cancel = float(np.mean(cancellation)) if len(cancellation) else 0.0

        # Simple amplification proxy: final divergence relative to initial non-zero divergence.
        eps = 1e-12
        init_ref = max(float(divergence[1]) if len(divergence) > 1 else eps, eps)
        chaos_amp = final_div / init_ref

        return AnalysisSummary(
            mode=mode,
            max_divergence=max_div,
            final_divergence=final_div,
            relative_energy_drift=relative_energy_drift,
            max_momentum_drift=max_momentum_drift,
            mean_perturbation=mean_pert,
            mean_cancellation=mean_cancel,
            chaos_amplification=float(chaos_amp),
        )

    def philosophical_report(self, summary: AnalysisSummary) -> str:
        deterministic_macro = summary.final_divergence < 0.2 and summary.relative_energy_drift < 0.05
        uncertainty_amplified = summary.chaos_amplification > 5.0
        cancellation_effective = summary.mean_cancellation > 0.5
        branching_like = summary.mode == "D" and summary.max_divergence > 0.0

        lines: list[str] = []
        lines.append("=== Philosophical Analysis Layer ===")
        lines.append(f"Mode: {summary.mode}")
        lines.append("")

        lines.append("1) Deterministic macro behavior")
        lines.append(
            "- Emergent determinism appears strong."
            if deterministic_macro
            else "- Macro trajectories show notable drift or instability under perturbation."
        )

        lines.append("2) Microscopic uncertainty amplification")
        lines.append(
            "- Small fluctuations are amplified by chaotic dynamics."
            if uncertainty_amplified
            else "- Fluctuations remain limited; amplification is weak in this run."
        )

        lines.append("3) Cancellation stabilization")
        lines.append(
            "- Cancellation mechanisms significantly suppress net perturbation effects."
            if cancellation_effective
            else "- Cancellation was weak or absent; net perturbative effects persisted."
        )

        lines.append("4) Branching interpretation signal")
        lines.append(
            "- Branch divergence is present, qualitatively resembling many-worlds-style path separation."
            if branching_like
            else "- No strong branch-style separation signal in this configuration."
        )

        lines.append("5) Classical logical consistency")
        lines.append(
            "- The observed trajectory remains classically coherent despite micro-level perturbations."
            if deterministic_macro
            else "- Classical coherence weakens as perturbations propagate into macro observables."
        )

        lines.append("")
        lines.append("Scientific boundary:")
        lines.append("- Results are simulation-dependent and do not constitute evidence against modern quantum theory.")

        return "\n".join(lines)
