"""Knapsack problem definition (O3), Ising encoding and an exact numpy QAOA reference.

Only `build_qaoa_circuit` needs Qiskit (imported lazily), so everything else
can be tested without it.

Conventions (important, they match Qiskit):
  * item i  <->  qubit i  <->  classical bit i
  * Qiskit count strings are little-endian: the RIGHTMOST character is item 0.
  * x_i = (1 - Z_i) / 2, so bit 1 = item selected = Z eigenvalue -1.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------
@dataclass
class KnapsackInstance:
    name: str
    names: list
    weights: list
    values: list
    capacity: int

    @property
    def n(self) -> int:
        return len(self.weights)


def load_instance(path) -> KnapsackInstance:
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    items = d["items"]
    return KnapsackInstance(
        name=d.get("name", "knapsack"),
        names=[it["name"] for it in items],
        weights=[int(it["weight"]) for it in items],
        values=[int(it["value"]) for it in items],
        capacity=int(d["capacity"]),
    )


def evaluate_bits(inst: KnapsackInstance, bits):
    """bits[i] = 1 if item i is selected. Returns (weight, value, feasible)."""
    w = int(sum(b * wi for b, wi in zip(bits, inst.weights)))
    v = int(sum(b * vi for b, vi in zip(bits, inst.values)))
    return w, v, w <= inst.capacity


def decode_bitstring(bitstr: str):
    """Qiskit count key (little-endian) -> list of item bits, item 0 first."""
    s = bitstr.replace(" ", "")
    return [int(ch) for ch in s[::-1]]


def bits_to_items(inst: KnapsackInstance, bits) -> str:
    chosen = [inst.names[i] for i, b in enumerate(bits) if b]
    return "+".join(chosen) if chosen else "(none)"


def brute_force(inst: KnapsackInstance) -> dict:
    """Exact classical baseline over all 2^n subsets."""
    rows = []
    for bits in itertools.product([0, 1], repeat=inst.n):
        w, v, feas = evaluate_bits(inst, bits)
        rows.append((v, w, feas, bits))
    feasible = sorted([r for r in rows if r[2]], key=lambda r: (-r[0], r[1]))
    best_v, best_w, _, best_bits = feasible[0]
    n_opt = sum(1 for r in feasible if r[0] == best_v)
    return {
        "best_value": int(best_v),
        "best_weight": int(best_w),
        "best_bits": list(best_bits),
        "best_items": bits_to_items(inst, best_bits),
        "n_optimal_solutions": int(n_opt),
        "n_feasible": len(feasible),
        "n_total": len(rows),
        "top5": [
            {"items": bits_to_items(inst, r[3]), "weight": int(r[1]), "value": int(r[0])}
            for r in feasible[:5]
        ],
    }


# --------------------------------------------------------------------------
# Cost function: unbalanced penalisation (no slack qubits)
#   cost(x) = -sum v_i x_i + lam1 * (C - W(x)) + lam2 * (C - W(x))^2
# --------------------------------------------------------------------------
def penalty_cost(inst: KnapsackInstance, bits, lam1: float, lam2: float) -> float:
    w, v, _ = evaluate_bits(inst, bits)
    s = inst.capacity - w
    return -v + lam1 * s + lam2 * s * s


def cost_diagonal(inst: KnapsackInstance, lam1: float, lam2: float) -> np.ndarray:
    """cost for every basis state; index bit i (little-endian) = item i."""
    n = inst.n
    out = np.zeros(2 ** n)
    for idx in range(2 ** n):
        bits = [(idx >> i) & 1 for i in range(n)]
        out[idx] = penalty_cost(inst, bits, lam1, lam2)
    return out


def ising_coefficients(inst: KnapsackInstance, lam1: float, lam2: float, scale: float = 1.0):
    """Return (offset, h, J) with cost = offset + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

    Everything is multiplied by `scale`.
    """
    n, w, v, c = inst.n, inst.weights, inst.values, inst.capacity
    const = lam1 * c + lam2 * c * c
    a = [-v[i] - lam1 * w[i] - 2 * lam2 * c * w[i] + lam2 * w[i] ** 2 for i in range(n)]
    b = {(i, j): 2 * lam2 * w[i] * w[j] for i in range(n) for j in range(i + 1, n)}
    offset = const + sum(a[i] / 2 for i in range(n)) + sum(bij / 4 for bij in b.values())
    h = [-a[i] / 2 - sum(b[tuple(sorted((i, j)))] / 4 for j in range(n) if j != i) for i in range(n)]
    J = {k: bij / 4 for k, bij in b.items()}
    return (
        float(scale * offset),
        [float(scale * x) for x in h],
        {k: float(scale * x) for k, x in J.items()},
    )


def ising_diagonal(n: int, offset: float, h, J) -> np.ndarray:
    out = np.zeros(2 ** n)
    for idx in range(2 ** n):
        z = [1 - 2 * ((idx >> i) & 1) for i in range(n)]
        val = offset + sum(h[i] * z[i] for i in range(n))
        val += sum(J[(i, j)] * z[i] * z[j] for (i, j) in J)
        out[idx] = val
    return out


def make_encoding(inst: KnapsackInstance, lam1: float = 0.0, lam2: float = 2.0, scale: float = 0.1) -> dict:
    """Bundle everything the rest of the pipeline needs."""
    diag = cost_diagonal(inst, lam1, lam2)
    offset, h, J = ising_coefficients(inst, lam1, lam2, scale=scale)
    check = ising_diagonal(inst.n, offset, h, J)
    err = float(np.max(np.abs(check - diag * scale)))
    if err > 1e-9:
        raise ValueError(f"Ising encoding does not match the cost function (max err {err:.3e})")
    return {
        "lam1": lam1, "lam2": lam2, "scale": scale,
        "offset": offset, "h": h, "J": J,
        "diag_raw": diag, "diag_scaled": diag * scale,
        "ground_index": int(np.argmin(diag)),
    }


# --------------------------------------------------------------------------
# Exact numpy QAOA (parameter optimisation + ideal reference distribution)
# --------------------------------------------------------------------------
def _apply_rx_all(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """exp(-i beta X) on every qubit (equals RX(2*beta))."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    t = psi.reshape([2] * n)
    for ax in range(n):
        a0 = np.take(t, 0, axis=ax)
        a1 = np.take(t, 1, axis=ax)
        t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return t.reshape(-1)


def qaoa_state(diag_scaled: np.ndarray, n: int, gammas, betas) -> np.ndarray:
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * diag_scaled)
        psi = _apply_rx_all(psi, n, b)
    return psi


def qaoa_probabilities(diag_scaled, n, gammas, betas) -> np.ndarray:
    return np.abs(qaoa_state(diag_scaled, n, gammas, betas)) ** 2


def optimize_qaoa(enc: dict, n: int, p: int, restarts: int = 40, seed: int = 7,
                  maxiter: int = 500, objective: str = "prob_opt") -> dict:
    """Offline parameter tuning on the exact (ideal) numpy simulator.

    objective = "prob_opt"  -> maximise the probability of the optimal bitstring
    objective = "energy"    -> minimise <C> (the textbook QAOA objective)

    The tuned parameters are then reused UNCHANGED on Processor A and B, so the
    comparison only reflects the architecture, not the optimiser.
    """
    diag = enc["diag_scaled"]
    gidx = enc["ground_index"]
    rng = np.random.default_rng(seed)

    def f(params):
        pr = qaoa_probabilities(diag, n, params[:p], params[p:])
        return -float(pr[gidx]) if objective == "prob_opt" else float(pr @ diag)

    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        if best is None or res.fun < best.fun:
            best = res
    params = np.asarray(best.x, dtype=float)
    pr = qaoa_probabilities(diag, n, params[:p], params[p:])
    return {
        "p": int(p),
        "objective": objective,
        "gammas": [float(x) for x in params[:p]],
        "betas": [float(x) for x in params[p:]],
        "ideal_prob_optimal": float(pr[gidx]),
        "ideal_probs": pr,
    }


# --------------------------------------------------------------------------
# Qiskit circuit
# --------------------------------------------------------------------------
def build_qaoa_circuit(enc: dict, n: int, gammas, betas, measure: bool = True):
    """Logical QAOA circuit. Cost layer: RZZ(2*gamma*J_ij) for ALL pairs + RZ(2*gamma*h_i)."""
    from qiskit import QuantumCircuit  # lazy import

    qc = QuantumCircuit(n, name="knapsack_qaoa")
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (i, j), jij in enc["J"].items():
            qc.rzz(2 * g * jij, i, j)
        for i in range(n):
            qc.rz(2 * g * enc["h"][i], i)
        for i in range(n):
            qc.rx(2 * b, i)
    if measure:
        qc.measure_all()
    return qc
