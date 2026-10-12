"""Circuit metrics, knapsack solution metrics, tables and automatic report text."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .problem import KnapsackInstance, bits_to_items, decode_bitstring, evaluate_bits, penalty_cost

TWO_QUBIT_NAMES = {"cx", "cz", "ecr", "swap", "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz"}


# ---------------------------------------------------------------- circuit metrics
def circuit_metrics(qc) -> dict:
    ops = dict(qc.count_ops())
    two_q = sum(v for k, v in ops.items() if k.lower() in TWO_QUBIT_NAMES)
    return {
        "depth": int(qc.depth()),
        "width_qubits": int(qc.num_qubits),
        "total_ops": int(sum(v for k, v in ops.items() if k not in ("barrier",))),
        "two_qubit_gates": int(two_q),
        "swap_count": int(ops.get("swap", 0)),
        "operations": {k: int(v) for k, v in ops.items()},
    }


# ---------------------------------------------------------------- statistics
def std_error(p: float, n: int) -> float:
    """Binomial standard error sqrt(p(1-p)/N)."""
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def hellinger_fidelity(counts: dict, ideal_probs: np.ndarray, n: int) -> float:
    """(sum_x sqrt(p_x q_x))^2 between sampled counts and the exact ideal distribution."""
    total = sum(counts.values())
    bc = 0.0
    for key, c in counts.items():
        bits = decode_bitstring(key)
        idx = sum(b << i for i, b in enumerate(bits))
        bc += math.sqrt((c / total) * float(ideal_probs[idx]))
    return float(bc ** 2)


# ---------------------------------------------------------------- solution metrics
def solution_metrics(counts: dict, inst: KnapsackInstance, classical: dict, enc: dict,
                     ideal_probs: np.ndarray) -> dict:
    shots = int(sum(counts.values()))
    opt_value = classical["best_value"]
    p_opt = p_feas = 0
    cost_sum = 0.0
    feas_value_sum = 0.0
    best = None  # (value, weight, bits)
    for key, c in counts.items():
        bits = decode_bitstring(key)
        w, v, feas = evaluate_bits(inst, bits)
        cost_sum += c * penalty_cost(inst, bits, enc["lam1"], enc["lam2"])
        if feas:
            p_feas += c
            feas_value_sum += c * v
            if best is None or v > best[0]:
                best = (v, w, bits)
            if v == opt_value:
                p_opt += c
    p_opt /= shots
    p_feas /= shots
    best_value = best[0] if best else 0
    return {
        "shots": shots,
        "prob_optimal": p_opt,
        "prob_optimal_se": std_error(p_opt, shots),
        "prob_feasible": p_feas,
        "prob_feasible_se": std_error(p_feas, shots),
        "best_feasible_value": int(best_value),
        "best_feasible_items": bits_to_items(inst, best[2]) if best else "(none)",
        "best_found_is_optimal": bool(best_value == opt_value),
        "approx_ratio_best": best_value / opt_value,
        "mean_value_feasible": (feas_value_sum / (p_feas * shots)) if p_feas > 0 else 0.0,
        "approx_ratio_mean_feasible": ((feas_value_sum / (p_feas * shots)) / opt_value) if p_feas > 0 else 0.0,
        "mean_penalized_cost": cost_sum / shots,
        "hellinger_fidelity_vs_ideal": hellinger_fidelity(counts, ideal_probs, inst.n),
    }


# ---------------------------------------------------------------- aggregation
def aggregate(rows: list[dict]) -> dict:
    """mean and std over a list of flat numeric dicts (one per transpiler seed)."""
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = float(df[col].mean())
            out[col + "_std"] = float(df[col].std(ddof=0))
    return out


def flat_row(cm: dict, sm: dict | None = None) -> dict:
    row = {k: v for k, v in cm.items() if k != "operations" and not isinstance(v, (dict, list))}
    if sm:
        row.update({k: v for k, v in sm.items() if not isinstance(v, (dict, list, str))})
    return row


def save_table(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- comparison helper
def compare(name: str, a: float, b: float, se_a: float = 0.0, se_b: float = 0.0,
            higher_is_better: bool = True, unit: str = "") -> str:
    """One data-driven sentence about A vs B that respects statistical uncertainty."""
    diff = b - a
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    fmt = lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:.0f}"
    if se > 0 and abs(diff) < 2 * se:
        return (f"{name}: A = {fmt(a)}{unit}, B = {fmt(b)}{unit}; the difference ({fmt(diff)}) "
                f"is within about 2 standard errors, so it is not statistically distinguishable.")
    if abs(diff) < 1e-12:
        return f"{name}: identical on A and B ({fmt(a)}{unit})."
    better = "B" if ((diff > 0) == higher_is_better) else "A"
    return (f"{name}: A = {fmt(a)}{unit}, B = {fmt(b)}{unit}; Processor {better} is better "
            f"(difference {fmt(abs(diff))}{unit}).")


# ---------------------------------------------------------------- report generation
def build_report(ctx: dict) -> str:
    """Markdown report using only measured numbers stored in ctx."""
    inst = ctx["inst"]
    cl = ctx["classical"]
    A, B = ctx["agg"]["A"], ctx["agg"]["B"]
    EA, EB = ctx["agg_equal"]["A"], ctx["agg_equal"]["B"]
    qa = ctx["qaoa"]
    cfg = ctx["config"]
    ph = [n for n in ("A", "B") if ctx["proc"][n].get("placeholder")]

    L = []
    L.append("# Phase 1 Report - O3 Knapsack\n")
    if ph:
        L.append(f"> **WARNING:** Processor(s) {', '.join(ph)} use PLACEHOLDER definitions, not the official "
                 "Challenge Kit files. Replace `processors/processor_*.json` and re-run the notebook "
                 "before submitting.\n")
    L.append("## 1. Problem\n")
    items = "; ".join(f"{n} (w={w}, v={v})" for n, w, v in zip(inst.names, inst.weights, inst.values))
    L.append(f"Selected Problem Statement: **O3 - Knapsack** (track: Quantum Optimization). "
             f"Capacity = {inst.capacity}. Items: {items}.\n")
    L.append(f"Exact classical baseline (brute force over {cl['n_total']} subsets): optimum = "
             f"**{cl['best_items']}**, weight {cl['best_weight']}, value **{cl['best_value']}** "
             f"({cl['n_optimal_solutions']} optimal solution(s), {cl['n_feasible']} feasible subsets).\n")
    L.append("## 2. Method\n")
    L.append(f"* Encoding: one qubit per item ({inst.n} qubits), unbalanced penalization with no slack qubits: "
             f"cost(x) = -value(x) + {qa['lam1']}*(C-W) + {qa['lam2']}*(C-W)^2, mapped to an Ising Hamiltonian "
             f"(all-to-all ZZ couplings because every item has non-zero weight).")
    L.append(f"* Algorithm: QAOA with p = {qa['p']} layers. Parameters were tuned offline on an exact numpy "
             f"simulator (objective: {qa['objective']}) and reused unchanged on both processors. "
             f"Ideal probability of the optimum = {qa['ideal_prob_optimal']:.3f} "
             f"(random guessing = {1 / 2 ** inst.n:.3f}).")
    L.append(f"* Controlled benchmark: same circuit, {cfg['shots']} shots, transpiler optimization level "
             f"{cfg['opt_level']}, {len(cfg['seeds'])} transpiler seeds ({cfg['seeds']}); values are mean over seeds "
             f"(+/- std over seeds).")
    for n_ in ("A", "B"):
        note = ctx["proc"][n_].get("noise_note")
        if note:
            L.append(f"* Processor {n_} noise assumption: 1q = {ctx['proc'][n_]['noise']['one_qubit_error']}, "
                     f"2q = {ctx['proc'][n_]['noise']['two_qubit_error']}, readout = {ctx['proc'][n_]['noise']['readout_error']}. {note}")
    L.append("* Noise: depolarizing 1q/2q errors and readout error from each processor JSON; plus an "
             "equal-noise control (both processors given Processor A's error rates) to separate "
             "connectivity effects from noise effects.\n")

    L.append("## 3. Results (A vs B, official noise)\n")
    def pm(d, k, f="{:.3f}"):
        return (f + " +/- " + f).format(d[k], d.get(k + "_std", 0.0))
    rows = [
        ("Circuit depth", "depth", "{:.1f}"),
        ("Two-qubit gates (CX)", "two_qubit_gates", "{:.1f}"),
        ("SWAPs (routing stage)", "swap_count", "{:.1f}"),
        ("Physical qubits used", "used_physical_qubits", "{:.1f}"),
        ("P(optimal) ideal", "ideal_prob_optimal", "{:.3f}"),
        ("P(optimal) noisy", "noisy_prob_optimal", "{:.3f}"),
        ("P(feasible) noisy", "noisy_prob_feasible", "{:.3f}"),
        ("Best sampled value / optimum (noisy)", "noisy_approx_ratio_best", "{:.3f}"),
        ("Approx. ratio of mean feasible (noisy)", "noisy_approx_ratio_mean_feasible", "{:.3f}"),
        ("Hellinger fidelity vs ideal (noisy)", "noisy_hellinger_fidelity_vs_ideal", "{:.3f}"),
        ("Transpile time (s)", "compile_seconds", "{:.4f}"),
        ("Noisy run time (s)", "runtime_noisy_seconds", "{:.4f}"),
    ]
    L.append("| Metric | Processor A | Processor B |")
    L.append("|---|---|---|")
    for label, key, f in rows:
        L.append(f"| {label} | {pm(A, key, f)} | {pm(B, key, f)} |")
    L.append("")
    L.append(f"Best feasible solution sampled (noisy): A -> {ctx['best_items']['A']}, B -> {ctx['best_items']['B']} "
             f"(optimum = {cl['best_items']}).\n")

    L.append("## 4. Interpretation\n")
    n_shots = cfg["shots"]
    L.append("**What changed?** " + compare("Two-qubit gates", A["two_qubit_gates"], B["two_qubit_gates"],
                                              higher_is_better=False))
    L.append("")
    L.append("**Routing.** " + compare("SWAPs", A["swap_count"], B["swap_count"], higher_is_better=False) +
             f" The QAOA cost layer needs all {inst.n * (inst.n - 1) // 2} pairwise ZZ interactions per layer, "
             "so the logical interaction graph is complete (K" + str(inst.n) + "); a sparse coupling graph can "
             "only realise it by inserting SWAPs.")
    L.append("")
    L.append("**Solution quality (official noise).** " +
             compare("P(optimal)", A["noisy_prob_optimal"], B["noisy_prob_optimal"],
                     A["noisy_prob_optimal_se"], B["noisy_prob_optimal_se"], True))
    L.append("")
    L.append("**Equal-noise control (connectivity only).** " +
             compare("P(optimal) with identical error rates", EA["noisy_prob_optimal"], EB["noisy_prob_optimal"],
                     EA["noisy_prob_optimal_se"], EB["noisy_prob_optimal_se"], True) +
             f" Two-qubit gates in this control: A = {EA['two_qubit_gates']:.1f}, B = {EB['two_qubit_gates']:.1f}.")
    L.append("")
    _keys = ("one_qubit_error", "two_qubit_error", "readout_error")
    same_noise = all(ctx["proc"]["A"]["noise"].get(k) == ctx["proc"]["B"]["noise"].get(k) for k in _keys)
    if same_noise:
        L.append("**Could gate-set / noise differences also contribute?** Not in this benchmark: both processors "
                 "use the same gate set (RZ, SX, X, CX) and the same error rates, so the equal-noise control is "
                 "identical to the official run by construction. Any A-vs-B difference therefore comes from the "
                 "coupling graph and the circuit the compiler produced for it. Noise still matters indirectly: "
                 "every extra CX gate is one more noisy operation.")
    else:
        L.append("**Could gate-set / noise differences also contribute?** Yes. The official runs combine a "
                 "different coupling graph AND different error rates. The equal-noise control removes the "
                 "noise difference, so any gap that remains there is due to connectivity (and the resulting "
                 "compiled circuit), while the change between the official and equal-noise results is due to the "
                 "noise assumptions.")
    L.append("")
    L.append("## 5. QAOA depth sweep (cost vs quality)\n")
    L.append("| p | ideal P(opt) | A: CX | A: noisy P(opt) | B: CX | B: noisy P(opt) |")
    L.append("|---|---|---|---|---|---|")
    for r in ctx["depth_sweep"]:
        L.append(f"| {r['p']} | {r['ideal_prob_optimal']:.3f} | {r['A_two_qubit_gates']:.1f} | "
                 f"{r['A_noisy_prob_optimal']:.3f} | {r['B_two_qubit_gates']:.1f} | {r['B_noisy_prob_optimal']:.3f} |")
    L.append("")
    L.append("## 6. Bottlenecks and Phase 2 readiness\n")
    ref = ctx.get("reference")
    if ref:
        L.append(f"* Routing overhead: the all-to-all reference circuit needs {ref['two_qubit_gates']:.0f} two-qubit gates; "
                 f"Processor A needs {A['two_qubit_gates']:.1f} (+{A['two_qubit_gates'] - ref['two_qubit_gates']:.1f}) and "
                 f"Processor B needs {B['two_qubit_gates']:.1f} (+{B['two_qubit_gates'] - ref['two_qubit_gates']:.1f}).")
    L.append(f"* Main bottleneck: the knapsack cost layer is a complete interaction graph on {inst.n} qubits; every missing "
             "coupling must be bridged with SWAPs, and each SWAP costs three CX gates that are exposed to noise.")
    L.append("* Phase 2 implication: a processor graph with a denser local neighbourhood (more cycles or higher degree "
             "among the qubits that are used) would cut routing overhead for this workload; extra qubits that are "
             "far from the used region do not help.")
    L.append("* Next experiments: a placement/routing study (initial layout choice) and a noise-aware variant that "
             "stops at a lower QAOA depth when the SWAP overhead makes deeper circuits unprofitable.\n")
    L.append("## 7. Limitations\n")
    L.append("* One small instance (5 items) on two finite processor models; this does not establish a "
             "universal architecture advantage.")
    L.append("* The Aer simulator with a simple depolarizing + readout noise model is not real hardware.")
    L.append("* Routing quality depends on the transpiler heuristic and seed; we report mean +/- std over seeds.")
    L.append("* QAOA parameters were tuned on an ideal simulator, not re-tuned under noise.")
    L.append("* SWAP counts come from a routing-only transpile with the same seed; the executed circuit may "
             "merge or cancel some SWAP-related CX gates.")
    L.append("* Statistical uncertainty is binomial (sqrt(p(1-p)/N)) per run plus the spread over transpiler seeds.\n")
    L.append("## 8. AI assistance disclosure\n")
    L.append("Parts of this repository were produced with AI assistance (Claude). The team reviewed, ran and "
             "takes responsibility for the submitted work.\n")
    return "\n".join(L)


def update_readme_results(readme_path, results_md: str) -> None:
    """Replace the block between the RESULTS markers in README.md."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        text += f"\n{start}\n{end}\n"
    pre = text.split(start)[0]
    post = text.split(end)[1]
    path.write_text(f"{pre}{start}\n{results_md}\n{end}{post}", encoding="utf-8")


# ---------------------------------------------------------------- one flat row per (processor, seed)
def case_row(result: dict, inst: KnapsackInstance, classical: dict, enc: dict, ideal_probs: np.ndarray) -> dict:
    """Flatten circuit metrics + ideal/noisy solution metrics (numeric only, prefixed)."""
    row = flat_row(result["circuit_metrics"])
    for tag, key in (("ideal", "ideal_counts"), ("noisy", "noisy_counts")):
        sm = solution_metrics(result[key], inst, classical, enc, ideal_probs)
        for k, v in sm.items():
            if isinstance(v, (int, float, bool)):
                row[f"{tag}_{k}"] = float(v)
    return row


def df_to_markdown(df: pd.DataFrame) -> str:
    """Small markdown-table writer (no extra dependency such as tabulate)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
