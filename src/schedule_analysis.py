"""Schedule aggregation and plotting helpers for CRH-CFG experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def _as_bool(value: object) -> bool:
    """Parse serialized boolean values without treating the string ``False`` as true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def aggregate_schedule_rows(rows: Iterable[dict]) -> list[Dict[str, float]]:
    """Aggregate per-sample controller rows into per-progress summary statistics."""
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["tau"])].append(row)
    summaries = []
    for tau in sorted(grouped):
        group = grouped[tau]
        scales = np.asarray([float(row["scale"]) for row in group], dtype=np.float64)
        margins = np.asarray([float(row["target_margin"]) for row in group], dtype=np.float64)
        drift = np.asarray([float(row["protected_drift"]) for row in group], dtype=np.float64)
        summaries.append({
            "tau": tau,
            "scale_mean": float(scales.mean()),
            "scale_q25": float(np.quantile(scales, 0.25)),
            "scale_median": float(np.quantile(scales, 0.5)),
            "scale_q75": float(np.quantile(scales, 0.75)),
            "target_margin_mean": float(margins.mean()),
            "protected_drift_mean": float(drift.mean()),
            "feasible_rate": sum(_as_bool(row["feasible"]) for row in group) / len(group),
            "fallback_rate": sum(_as_bool(row["fallback_selected"]) for row in group) / len(group),
        })
    return summaries


def write_schedule_summary(rows: Iterable[dict], path: str | Path) -> None:
    """Write per-progress schedule summaries as a CSV artifact."""
    summaries = aggregate_schedule_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        path.write_text("tau\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def plot_schedule_summary(summary_csv: str | Path, output_path: str | Path) -> None:
    """Plot scale, feasibility, target margin, and protected drift over ``tau``."""
    import matplotlib.pyplot as plt

    with Path(summary_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No schedule rows found in {summary_csv}.")
    tau = np.asarray([float(row["tau"]) for row in rows])
    mean_scale = np.asarray([float(row["scale_mean"]) for row in rows])
    q25 = np.asarray([float(row["scale_q25"]) for row in rows])
    q75 = np.asarray([float(row["scale_q75"]) for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes[0, 0].plot(tau, mean_scale, label="mean w")
    axes[0, 0].fill_between(tau, q25, q75, alpha=0.25, label="IQR")
    axes[0, 0].set_ylabel("CFG scale")
    axes[0, 0].legend()
    axes[0, 1].plot(tau, [float(row["feasible_rate"]) for row in rows], label="feasible")
    axes[0, 1].plot(tau, [float(row["fallback_rate"]) for row in rows], label="fallback")
    axes[0, 1].set_ylabel("Decision rate")
    axes[0, 1].legend()
    axes[1, 0].plot(tau, [float(row["target_margin_mean"]) for row in rows])
    axes[1, 0].set_ylabel("Target margin")
    axes[1, 1].plot(tau, [float(row["protected_drift_mean"]) for row in rows])
    axes[1, 1].set_ylabel("Protected drift")
    for axis in axes.flat:
        axis.set_xlabel("Generation progress tau")
        axis.grid(alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
