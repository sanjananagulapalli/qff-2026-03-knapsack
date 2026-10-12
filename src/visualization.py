"""Figures for the Phase 1 A/B comparison (matplotlib + networkx)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .problem import bits_to_items, penalty_cost
from .processors import coupling_graph

COL_A = "#1f6feb"
COL_B = "#2ea043"
COL_REF = "#8b949e"


def _finish(fig, path, show=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_processor_graph(proc: dict, path, color=COL_A, show=True):
    g = coupling_graph(proc)
    pos = nx.kamada_kawai_layout(g) if g.number_of_nodes() > 2 else nx.spring_layout(g, seed=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    nx.draw_networkx_edges(g, pos, ax=ax, width=2, edge_color="#555")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=color, node_size=700, edgecolors="white")
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_weight="bold")
    ax.set_title(f"{proc['name']}: {proc['num_qubits']} qubits, {g.number_of_edges()} couplings")
    ax.axis("off")
    return _finish(fig, path, show)


def draw_circuit(qc, path, show=True, fold=60):
    """Try Qiskit's matplotlib drawer (needs pylatexenc); fall back to a text rendering."""
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
        fig, ax = plt.subplots(figsize=(14, max(3, 0.18 * (txt.count("\n") + 2))))
        ax.text(0.0, 1.0, txt, family="monospace", fontsize=5, va="top")
        ax.axis("off")
        print(f"(mpl circuit drawer unavailable: {type(exc).__name__}; saved text rendering instead)")
        return _finish(fig, path, show)


def plot_logical_vs_transpiled(rows: dict, path, show=True):
    """rows = {label: {'depth':..., 'two_qubit_gates':..., 'swap_count':...}}"""
    labels = list(rows)
    colors = [COL_REF, COL_A, COL_B][: len(labels)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title in zip(axes, ("depth", "two_qubit_gates", "swap_count"),
                              ("Circuit depth", "Two-qubit gates (CX)", "SWAPs inserted")):
        vals = [rows[l][key] for l in labels]
        errs = [rows[l].get(key + "_std", 0.0) for l in labels]
        ax.bar(labels, vals, yerr=errs, color=colors, capsize=3)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Logical (all-to-all) vs transpiled circuits", y=1.02)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_ab_performance(agg: dict, agg_equal: dict, path, show=True):
    """Grouped bars of P(opt), P(feasible), Hellinger fidelity: A/B official and equal-noise."""
    metrics = [("noisy_prob_optimal", "P(optimal)"),
               ("noisy_prob_feasible", "P(feasible)"),
               ("noisy_hellinger_fidelity_vs_ideal", "Hellinger fidelity")]
    groups = [("A", agg["A"], COL_A, ""), ("B", agg["B"], COL_B, ""),
              ("A equal-noise", agg_equal["A"], COL_A, "//"), ("B equal-noise", agg_equal["B"], COL_B, "//")]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (lab, d, col, hatch) in enumerate(groups):
        vals = [d[k] for k, _ in metrics]
        errs = [d.get(k + "_std", 0.0) for k, _ in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, color=col, hatch=hatch, alpha=0.9 if not hatch else 0.55,
               label=lab, capsize=2, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value (mean +/- std over transpiler seeds)")
    ax.set_title("Processor A vs B: noisy performance (hatched = equal-noise control)")
    ax.legend(ncol=2, fontsize=8)
    return _finish(fig, path, show)


def plot_distribution(inst, enc, ideal_probs, counts_a, counts_b, path, show=True):
    """Probability of every bitstring, ordered by penalised cost (best first)."""
    n = inst.n
    order = sorted(range(2 ** n), key=lambda idx: penalty_cost(
        inst, [(idx >> i) & 1 for i in range(n)], enc["lam1"], enc["lam2"]))

    def probs(counts):
        out = np.zeros(2 ** n)
        total = sum(counts.values())
        for key, c in counts.items():
            bits = [int(ch) for ch in key.replace(" ", "")[::-1]]
            out[sum(b << i for i, b in enumerate(bits))] += c / total
        return out

    pa, pb = probs(counts_a), probs(counts_b)
    labels = [bits_to_items(inst, [(idx >> i) & 1 for i in range(n)]) for idx in order]
    x = np.arange(2 ** n)
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w, [ideal_probs[i] for i in order], w, color=COL_REF, label="ideal (exact)")
    ax.bar(x, [pa[i] for i in order], w, color=COL_A, label="Processor A (noisy)")
    ax.bar(x + w, [pb[i] for i in order], w, color=COL_B, label="Processor B (noisy)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("probability")
    ax.set_title("Outcome distribution, ordered by penalised cost (optimum on the left)")
    ax.legend()
    return _finish(fig, path, show)


def plot_tradeoffs(depth_sweep: list, path, show=True):
    """Noisy P(optimal) versus two-qubit gate count for each QAOA depth p."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for tag, col in (("A", COL_A), ("B", COL_B)):
        xs = [r[f"{tag}_two_qubit_gates"] for r in depth_sweep]
        ys = [r[f"{tag}_noisy_prob_optimal"] for r in depth_sweep]
        ax.plot(xs, ys, "o-", color=col, label=f"Processor {tag}")
        for r, xv, yv in zip(depth_sweep, xs, ys):
            ax.annotate(f"p={r['p']}", (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(1 / 32, color=COL_REF, ls="--", label="random guess")
    ax.set_xlabel("two-qubit gates after compilation")
    ax.set_ylabel("noisy P(optimal)")
    ax.set_title("Quality vs circuit cost (QAOA depth sweep)")
    ax.legend()
    return _finish(fig, path, show)


def plot_landscape(gammas, betas, z_exact, z_noisy, path, label="P(optimal)", noisy_title="Processor A, noisy", show=True):
    """Two heat-maps of a QAOA p=1 payoff landscape (rows = beta, columns = gamma)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, z, title in ((axes[0], z_exact, "Exact (ideal) simulation"), (axes[1], z_noisy, noisy_title)):
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis",
                       extent=[gammas[0], gammas[-1], betas[0], betas[-1]])
        k = np.unravel_index(np.argmax(z), z.shape)
        ax.plot(np.linspace(gammas[0], gammas[-1], z.shape[1])[k[1]],
                np.linspace(betas[0], betas[-1], z.shape[0])[k[0]], "r*", ms=12)
        ax.set_xlabel("gamma")
        ax.set_ylabel("beta")
        ax.set_title(f"{title} (best = {z.max():.3f})")
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle("QAOA p=1 payoff landscape (red star = best angles)", y=1.02)
    fig.tight_layout()
    return _finish(fig, path, show)
