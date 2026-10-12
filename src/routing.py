"""Transpilation, routing analysis and execution on Processor A/B.

Three kinds of compiled circuit are used:
  * all-to-all reference : same basis gates, NO coupling constraint
  * routed (swap-visible): routing-only transpile whose basis keeps `swap`, so SWAPs can be counted
  * final compiled       : the circuit that is actually executed (processor basis gates)
All use the same optimisation level and transpiler seed.
"""
from __future__ import annotations

import time

import numpy as np

from .processors import build_coupling_map, build_noise_model

OPT_LEVEL = 1


def _transpile(circ, **kw):
    from qiskit import transpile

    return transpile(circ, optimization_level=OPT_LEVEL, **kw)


def compile_all_to_all(logical, proc: dict, seed: int):
    """Reference: decomposed to the processor basis but with full connectivity."""
    return _transpile(logical, basis_gates=list(proc["basis_gates"]), seed_transpiler=seed)


def compile_for_processor(logical, proc: dict, seed: int):
    """Return (compiled, routed_with_swaps, compile_seconds)."""
    cm = build_coupling_map(proc)
    t0 = time.perf_counter()
    compiled = _transpile(
        logical, coupling_map=cm, basis_gates=list(proc["basis_gates"]), seed_transpiler=seed
    )
    dt = time.perf_counter() - t0
    swap_basis = sorted(set(proc["basis_gates"]) | {"swap", "rzz", "rx", "h"})
    routed = _transpile(logical, coupling_map=cm, basis_gates=swap_basis, seed_transpiler=seed)
    return compiled, routed, dt


def used_physical_qubits(compiled) -> int:
    used = set()
    for inst in compiled.data:
        for q in inst.qubits:
            used.add(compiled.find_bit(q).index)
    return len(used)


def initial_layout(compiled):
    """logical->physical list if Qiskit exposes it, else None."""
    try:
        return list(compiled.layout.initial_index_layout(filter_ancillas=True))
    except Exception:
        return None


def run_counts(compiled, shots: int, seed_sim: int, noise_model=None) -> dict:
    """Execute on Aer (ideal if noise_model is None) and return counts."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
    t0 = time.perf_counter()
    result = sim.run(compiled, shots=shots, seed_simulator=seed_sim).result()
    runtime = time.perf_counter() - t0
    return dict(result.get_counts()), runtime


def run_case(logical, proc: dict, seed: int, shots: int, noise_override: dict | None = None):
    """Compile once, run ideal AND noisy. Returns a dict with everything needed."""
    from .metrics import circuit_metrics

    compiled, routed, t_compile = compile_for_processor(logical, proc, seed)
    cm = circuit_metrics(compiled)
    rm = circuit_metrics(routed)
    cm["swap_count"] = rm["swap_count"]
    cm["used_physical_qubits"] = used_physical_qubits(compiled)
    cm["compile_seconds"] = t_compile

    ideal_counts, t_ideal = run_counts(compiled, shots, seed_sim=seed)
    nm = build_noise_model(proc, override=noise_override)
    noisy_counts, t_noisy = run_counts(compiled, shots, seed_sim=seed, noise_model=nm)
    cm["runtime_ideal_seconds"] = t_ideal
    cm["runtime_noisy_seconds"] = t_noisy
    return {
        "compiled": compiled,
        "routed": routed,
        "circuit_metrics": cm,
        "ideal_counts": ideal_counts,
        "noisy_counts": noisy_counts,
        "layout": initial_layout(compiled),
    }


def compile_and_run_noisy(logical, proc: dict, seed: int, shots: int, noise_model):
    """Light-weight helper for landscape scans: one transpile + one noisy run -> counts."""
    cm = build_coupling_map(proc)
    compiled = _transpile(logical, coupling_map=cm, basis_gates=list(proc["basis_gates"]), seed_transpiler=seed)
    counts, _ = run_counts(compiled, shots, seed_sim=seed, noise_model=noise_model)
    return counts
