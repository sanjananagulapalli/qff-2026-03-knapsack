
"""Figures for the Qubit Crew Phase 2 processor comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .problem import bits_to_items, penalty_cost
from .processors import coupling_graph

COL_A = "#1f6feb"
COL_B = "#2ea043"
COL_D = "#d97706"
COL_REF = "#8b949e"

PROCESSOR_COLORS = {
    "A": COL_A,
    "B": COL_B,
    "D": COL_D,
}


def _finish(fig, path, show=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_processor_graph(proc: dict, path, color=None, show=True):
    """Draw every physical qubit and coupling in a processor."""
    g = coupling_graph(proc)
    name = str(proc["name"])
    tag = next(
        (t for t in PROCESSOR_COLORS if f"PROCESSOR {t}" in name.upper()),
        None,
    )

    if tag is None and name.strip().upper() in PROCESSOR_COLORS:
        tag = name.strip().upper()

    if color is None:
        color = PROCESSOR_COLORS.get(tag, COL_A)

    pos = (
        nx.kamada_kawai_layout(g)
        if g.number_of_nodes() > 2
        else nx.spring_layout(g, seed=1)
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(
        g, pos, ax=ax, width=2, edge_color="#555555"
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=color,
        node_size=700, edgecolors="white"
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax, font_color="white", font_weight="bold"
    )
    ax.set_title(
        f"{name}: {proc['num_qubits']} qubits, "
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
    except Exception as exc:
        txt = str(qc.draw("text", fold=fold, idle_wires=False))
        fig, ax = plt.subplots(
            figsize=(14, max(3, 0.18 * (txt.count("\n") + 2)))
        )
        ax.text(
            0.0, 1.0, txt, family="monospace",
            fontsize=5, va="top"
        )
        ax.axis("off")
        print(
            f"(mpl circuit drawer unavailable: "
            f"{type(exc).__name__}; saved text rendering instead)"
        )
        return _finish(fig, path, show)


def plot_logical_vs_transpiled(rows: dict, path, show=True):
    """Compare circuit depth, two-qubit gate count and inserted SWAPs."""
    if not rows:
        raise ValueError("At least one circuit-metrics row is required.")

    labels = list(rows)
    colors = [
        PROCESSOR_COLORS.get(
            str(label).strip().upper().replace("PROCESSOR ", ""),
            COL_REF,
        )
        for label in labels
    ]

    keys = ("depth", "two_qubit_gates", "swap_count")
    titles = ("Circuit depth", "Two-qubit gates", "SWAPs inserted")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, key, title in zip(axes, keys, titles):
        vals = [rows[label][key] for label in labels]
        errs = [
            rows[label].get(key + "_std", 0.0)
            for label in labels
        ]
        ax.bar(
            labels, vals, yerr=errs, color=colors, capsize=3
        )
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)

        for i, value in enumerate(vals):
            ax.text(
                i, value, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9
            )

    fig.suptitle("Logical vs transpiled circuit metrics", y=1.02)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_ab_performance(agg: dict, agg_equal: dict, path, show=True):
    """Keep the original A/B official-noise and equal-noise comparison."""
    metrics = [
        ("noisy_prob_optimal", "P(optimal)"),
        ("noisy_prob_feasible", "P(feasible)"),
        ("noisy_hellinger_fidelity_vs_ideal", "Hellinger fidelity"),
    ]

    groups = [
        ("A", agg["A"], COL_A, ""),
        ("B", agg["B"], COL_B, ""),
        ("A equal-noise", agg_equal["A"], COL_A, "//"),
        ("B equal-noise", agg_equal["B"], COL_B, "//"),
    ]

    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (label, data, color, hatch) in enumerate(groups):
        values = [data[key] for key, _ in metrics]
        errors = [
            data.get(key + "_std", 0.0)
            for key, _ in metrics
        ]
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            yerr=errors,
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
    ax.set_ylabel("Metric value")
    ax.set_title("Processors A/B: official noise vs equal-noise control")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_processor_performance(
    agg: dict, path, show=True, title=None
):
    """Compare noisy performance across whichever of A, B and D are present."""
    available = [
        tag for tag in ("A", "B", "D")
        if tag in agg
    ]
    if not available:
        raise ValueError("No processor results found for A, B or D.")

    metrics = [
        ("noisy_prob_optimal", "P(optimal)"),
        ("noisy_prob_feasible", "P(feasible)"),
        ("noisy_hellinger_fidelity_vs_ideal", "Hellinger fidelity"),
    ]

    x = np.arange(len(metrics))
    width = 0.75 / len(available)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, tag in enumerate(available):
        data = agg[tag]
        values = [data[key] for key, _ in metrics]
        errors = [
            data.get(key + "_std", 0.0)
            for key, _ in metrics
        ]
        offset = (i - (len(available) - 1) / 2) * width

        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            color=PROCESSOR_COLORS[tag],
            label=f"Processor {tag}",
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title(
        title or "Noisy performance: Processor A vs B vs D"
    )
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_distribution(
    inst, enc, ideal_probs, counts_a, counts_b,
    path, show=True, counts_d=None
):
    """Plot ideal and measured outcome probabilities."""
    n = inst.n
    number_of_states = 2 ** n

    ideal_probs = np.asarray(ideal_probs, dtype=float)
    if len(ideal_probs) != number_of_states:
        raise ValueError(
            f"Expected {number_of_states} ideal probabilities; "
            f"received {len(ideal_probs)}."
        )

    order = sorted(
        range(number_of_states),
        key=lambda idx: penalty_cost(
            inst,
            [(idx >> i) & 1 for i in range(n)],
            enc["lam1"],
            enc["lam2"],
        ),
    )

    def probabilities(counts):
        if not counts:
            raise ValueError("Measurement counts cannot be empty.")

        total = sum(counts.values())
        if total <= 0:
            raise ValueError("Measurement counts must sum to a positive value.")

        output = np.zeros(number_of_states)

        for key, count in counts.items():
            bitstring = key.replace(" ", "")
            if len(bitstring) != n or any(
                char not in "01" for char in bitstring
            ):
                raise ValueError(
                    f"Invalid {n}-qubit measurement bitstring: {key!r}"
                )
            if count < 0:
                raise ValueError("Measurement counts cannot be negative.")

            # Qiskit displays classical bit 0 on the right.
            bits = [int(char) for char in bitstring[::-1]]
            index = sum(bit << i for i, bit in enumerate(bits))
            output[index] += count / total

        return output

    distributions = [
        ("Processor A (noisy)", probabilities(counts_a), COL_A),
        ("Processor B (noisy)", probabilities(counts_b), COL_B),
    ]

    if counts_d is not None:
        distributions.append(
            ("Processor D (noisy)", probabilities(counts_d), COL_D)
        )

    labels = [
        bits_to_items(inst, [(idx >> i) & 1 for i in range(n)])
        for idx in order
    ]
    x = np.arange(number_of_states)
    width = 0.8 / (1 + len(distributions))

    fig, ax = plt.subplots(figsize=(max(12, number_of_states * 0.3), 5))
    ax.bar(
        x - width * len(distributions) / 2,
        [ideal_probs[idx] for idx in order],
        width,
        color=COL_REF,
        label="Ideal (exact)",
    )

    for i, (label, values, color) in enumerate(distributions):
        offset = width * (i + 1 - len(distributions) / 2)
        ax.bar(
            x + offset, [values[idx] for idx in order],
            width, color=color, label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Probability")
    ax.set_title("Outcome distribution, ordered by penalized cost")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_tradeoffs(depth_sweep: list, path, show=True):
    """Plot noisy optimal-solution probability against gate count."""
    if not depth_sweep:
        raise ValueError("The depth sweep contains no results.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False

    for tag in ("A", "B", "D"):
        x_key = f"{tag}_two_qubit_gates"
        y_key = f"{tag}_noisy_prob_optimal"

        valid_rows = [
            row for row in depth_sweep
            if x_key in row and y_key in row
        ]
        if not valid_rows:
            continue

        xs = [row[x_key] for row in valid_rows]
        ys = [row[y_key] for row in valid_rows]

        ax.plot(
            xs, ys, "o-",
            color=PROCESSOR_COLORS[tag],
            label=f"Processor {tag}",
        )

        for row, xv, yv in zip(valid_rows, xs, ys):
            if "p" in row:
                ax.annotate(
                    f"p={row['p']}",
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError(
            "No A/B/D gate-count and optimal-probability columns "
            "were found in depth_sweep."
        )

    ax.axhline(
        1 / 32, color=COL_REF, ls="--", label="Uniform baseline (5 bits)"
    )
    ax.set_xlabel("Two-qubit gates after compilation")
    ax.set_ylabel("Noisy P(optimal)")
    ax.set_title("Solution quality vs circuit cost")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_landscape(
    gammas, betas, z_exact, z_noisy, path,
    label="P(optimal)",
    noisy_title="Processor A, noisy",
    show=True,
):
    """Plot ideal and noisy QAOA p=1 parameter landscapes."""
    gammas = np.asarray(gammas)
    betas = np.asarray(betas)
    z_exact = np.asarray(z_exact)
    z_noisy = np.asarray(z_noisy)

    expected_shape = (len(betas), len(gammas))
    if z_exact.shape != expected_shape or z_noisy.shape != expected_shape:
        raise ValueError(
            "Landscape arrays must have shape "
            f"(len(betas), len(gammas)) = {expected_shape}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, values, title in (
        (axes[0], z_exact, "Exact (ideal) simulation"),
        (axes[1], z_noisy, noisy_title),
    ):
        im = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[
                gammas[0], gammas[-1],
                betas[0], betas[-1],
            ],
        )

        best = np.unravel_index(np.argmax(values), values.shape)
        best_gamma = gammas[best[1]]
        best_beta = betas[best[0]]

        ax.plot(best_gamma, best_beta, "r*", ms=12)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Beta")
        ax.set_title(f"{title} (best = {values.max():.3f})")
        fig.colorbar(im, ax=ax, label=label)

    fig.suptitle("QAOA p=1 parameter landscape", y=1.02)
    fig.tight_layout()
    return _finish(fig, path, show)
