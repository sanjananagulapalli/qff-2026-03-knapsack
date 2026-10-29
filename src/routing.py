
"""Figures for Phase 2 processor and QAOA comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .problem import bits_to_items, penalty_cost
from .processors import coupling_graph


COL_A = "#1f6feb"
COL_B = "#2ea043"
COL_D = "#d29922"
COL_REF = "#8b949e"

PROCESSOR_COLORS = {
    "A": COL_A,
    "B": COL_B,
    "D": COL_D,
}


def _finish(fig, path, show=True):
    """Save a figure, optionally display it, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _processor_tag(proc: dict) -> str:
    """Extract a processor tag such as A, B, or D."""
    name = str(proc.get("name", "")).upper()
    for tag in ("A", "B", "D"):
        if f"PROCESSOR {tag}" in name or name == tag:
            return tag
    return name or "Processor"


def _color_for(label: str) -> str:
    """Choose a consistent color for each processor."""
    text = str(label).upper()
    for tag, color in PROCESSOR_COLORS.items():
        if tag in text:
            return color
    return COL_REF


def plot_processor_graph(proc: dict, path, color=None, show=True):
    """Plot every physical qubit and coupling in a processor."""
    g = coupling_graph(proc)
    tag = _processor_tag(proc)
    color = color or PROCESSOR_COLORS.get(tag, COL_REF)

    if g.number_of_nodes() > 2:
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.spring_layout(g, seed=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    nx.draw_networkx_edges(
        g, pos, ax=ax, width=2, edge_color="#555555"
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=color,
        node_size=700,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax, font_color="white", font_weight="bold"
    )

    ax.set_title(
        f"{proc['name']}: {proc['num_qubits']} qubits, "
        f"{g.number_of_edges()} couplings"
    )
    ax.axis("off")
    return _finish(fig, path, show)


def draw_circuit(qc, path, show=True, fold=60):
    """Draw a Qiskit circuit, falling back to a text rendering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig = qc.draw("mpl", fold=fold, idle_wires=False)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    except Exception as exc:  # noqa: BLE001
        txt = str(qc.draw("text", fold=fold, idle_wires=False))
        fig, ax = plt.subplots(
            figsize=(14, max(3, 0.18 * (txt.count("\n") + 2)))
        )
        ax.text(
            0.0, 1.0, txt,
            family="monospace", fontsize=5, va="top"
        )
        ax.axis("off")
        print(
            f"(Matplotlib circuit drawer unavailable: "
            f"{type(exc).__name__}; saved text rendering instead)"
        )
        return _finish(fig, path, show)


def plot_logical_vs_transpiled(rows: dict, path, show=True):
    """Compare circuit depth, two-qubit gates, and inserted SWAPs."""
    if not rows:
        raise ValueError("rows must contain at least one circuit result.")

    labels = list(rows)
    colors = [_color_for(label) for label in labels]

    metrics = [
        ("depth", "Circuit depth"),
        ("two_qubit_gates", "Two-qubit gates"),
        ("swap_count", "SWAPs inserted"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (key, title) in zip(axes, metrics):
        vals = [float(rows[label].get(key, 0)) for label in labels]
        errs = [
            float(rows[label].get(key + "_std", 0.0))
            for label in labels
        ]

        bars = ax.bar(
            labels, vals, yerr=errs, color=colors, capsize=3
        )
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)

        for bar, value in zip(bars, vals):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle("Logical versus transpiled circuit costs")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_ab_performance(agg: dict, agg_equal: dict, path, show=True):
    """Compare A/B official-noise results and equal-noise controls."""
    metrics = [
        ("noisy_prob_optimal", "P(optimal)"),
        ("noisy_prob_feasible", "P(feasible)"),
        ("noisy_hellinger_fidelity_vs_ideal", "Hellinger fidelity"),
    ]

    groups = []

    for tag in ("A", "B"):
        if tag in agg:
            groups.append(
                (tag, agg[tag], PROCESSOR_COLORS[tag], "")
            )

    for tag in ("A", "B"):
        if tag in agg_equal:
            groups.append(
                (
                    f"{tag} equal-noise",
                    agg_equal[tag],
                    PROCESSOR_COLORS[tag],
                    "//",
                )
            )

    if not groups:
        raise ValueError("No A/B performance results were supplied.")

    x = np.arange(len(metrics))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (label, data, color, hatch) in enumerate(groups):
        vals = [float(data.get(key, 0.0)) for key, _ in metrics]
        errs = [
            float(data.get(key + "_std", 0.0))
            for key, _ in metrics
        ]
        offset = (i - (len(groups) - 1) / 2) * width

        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            color=color,
            hatch=hatch,
            alpha=0.9 if not hatch else 0.55,
            label=label,
            capsize=2,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value")
    ax.set_title("A/B performance: official noise and equal-noise control")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_processor_performance(
    results: dict, path, show=True, title=None
):
    """Compare A/B/D performance using aggregated processor metrics."""
    if not results:
        raise ValueError("results must contain processor metrics.")

    metrics = [
        ("prob_optimal", "noisy_prob_optimal", "P(optimal)"),
        ("prob_feasible", "noisy_prob_feasible", "P(feasible)"),
        (
            "hellinger_fidelity_vs_ideal",
            "noisy_hellinger_fidelity_vs_ideal",
            "Hellinger fidelity",
        ),
    ]

    tags = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / len(tags)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, tag in enumerate(tags):
        data = results[tag]
        values = []
        errors = []

        for plain_key, noisy_key, _ in metrics:
            key = noisy_key if noisy_key in data else plain_key
            values.append(float(data.get(key, 0.0)))
            errors.append(float(data.get(key + "_std", 0.0)))

        offset = (i - (len(tags) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            color=_color_for(tag),
            capsize=2,
            label=f"Processor {tag}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value")
    ax.set_title(title or "Processor A/B/D performance")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_distribution(
    inst,
    enc,
    ideal_probs,
    counts_a,
    counts_b,
    path,
    show=True,
    counts_d=None,
):
    """Plot ideal and sampled outcome distributions for A/B and optional D.

    Uses the Qiskit bitstring convention: the rightmost bit represents
    item/qubit 0.
    """
    n = inst.n
    num_states = 2 ** n

    if len(ideal_probs) != num_states:
        raise ValueError(
            f"Expected {num_states} ideal probabilities, "
            f"received {len(ideal_probs)}."
        )

    order = sorted(
        range(num_states),
        key=lambda idx: penalty_cost(
            inst,
            [(idx >> i) & 1 for i in range(n)],
            enc["lam1"],
            enc["lam2"],
        ),
    )

    def probs(counts):
        out = np.zeros(num_states, dtype=float)
        total = sum(counts.values())

        if total <= 0:
            raise ValueError(
                "Cannot plot an empty measurement-count dictionary."
            )

        for key, count in counts.items():
            bitstring = key.replace(" ", "")

            if len(bitstring) != n or any(
                char not in "01" for char in bitstring
            ):
                raise ValueError(
                    f"Invalid {n}-bit measurement key: {key!r}"
                )

            bits = [int(char) for char in bitstring[::-1]]
            index = sum(bit << i for i, bit in enumerate(bits))
            out[index] += count / total

        return out

    series = [
        ("Ideal (exact)", np.asarray(ideal_probs), COL_REF),
        ("Processor A (noisy)", probs(counts_a), COL_A),
        ("Processor B (noisy)", probs(counts_b), COL_B),
    ]

    if counts_d is not None:
        series.append(
            ("Processor D (noisy)", probs(counts_d), COL_D)
        )

    labels = [
        bits_to_items(inst, [(idx >> i) & 1 for i in range(n)])
        for idx in order
    ]

    x = np.arange(num_states)
    width = 0.8 / len(series)

    fig, ax = plt.subplots(
        figsize=(max(12, num_states * 0.28), 4.5)
    )

    for i, (label, probabilities, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(
            x + offset,
            [probabilities[idx] for idx in order],
            width,
            color=color,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Probability")
    ax.set_title("Outcome distribution ordered by penalized cost")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_tradeoffs(depth_sweep: list, path, show=True):
    """Plot noisy P(optimal) against compiled two-qubit gate count."""
    if not depth_sweep:
        raise ValueError("depth_sweep must contain at least one row.")

    tags = [
        tag for tag in ("A", "B", "D")
        if any(
            f"{tag}_two_qubit_gates" in row
            and f"{tag}_noisy_prob_optimal" in row
            for row in depth_sweep
        )
    ]

    if not tags:
        raise ValueError(
            "No A/B/D two-qubit gate metrics found in depth_sweep."
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for tag in tags:
        rows = [
            row for row in depth_sweep
            if f"{tag}_two_qubit_gates" in row
            and f"{tag}_noisy_prob_optimal" in row
        ]

        xs = [row[f"{tag}_two_qubit_gates"] for row in rows]
        ys = [row[f"{tag}_noisy_prob_optimal"] for row in rows]

        ax.plot(
            xs,
            ys,
            "o-",
            color=PROCESSOR_COLORS[tag],
            label=f"Processor {tag}",
        )

        for row, xv, yv in zip(rows, xs, ys):
            ax.annotate(
                f"p={row['p']}",
                (xv, yv),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.axhline(
        1 / 32,
        color=COL_REF,
        linestyle="--",
        label="Random baseline",
    )
    ax.set_xlabel("Two-qubit gates after compilation")
    ax.set_ylabel("Noisy P(optimal)")
    ax.set_title("Solution quality versus circuit cost")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_landscape(
    gammas,
    betas,
    z_exact,
    z_noisy,
    path,
    label="P(optimal)",
    noisy_title="Processor A, noisy",
    show=True,
):
    """Plot ideal and noisy QAOA p=1 payoff landscapes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, z, title in (
        (axes[0], z_exact, "Exact (ideal) simulation"),
        (axes[1], z_noisy, noisy_title),
    ):
        z = np.asarray(z)

        if z.ndim != 2:
            raise ValueError("Landscape values must be a 2D array.")

        if z.shape != (len(betas), len(gammas)):
            raise ValueError(
                "Landscape shape must be (len(betas), len(gammas))."
            )

        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[
                gammas[0],
                gammas[-1],
                betas[0],
                betas[-1],
            ],
        )

        best = np.unravel_index(np.argmax(z), z.shape)
        best_gamma = np.linspace(
            gammas[0], gammas[-1], z.shape[1]
        )[best[1]]
        best_beta = np.linspace(
            betas[0], betas[-1], z.shape[0]
        )[best[0]]

        ax.plot(best_gamma, best_beta, "r*", ms=12)
        ax.set_xlabel("gamma")
        ax.set_ylabel("beta")
        ax.set_title(f"{title} (best = {z.max():.3f})")
        fig.colorbar(im, ax=ax, label=label)

    fig.suptitle(
        "QAOA p=1 payoff landscape (red star = best angles)",
        y=1.02,
    )
    fig.tight_layout()
    return _finish(fig, path, show)
