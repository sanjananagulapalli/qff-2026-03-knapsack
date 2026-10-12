"""Processor A/B definitions -> Qiskit CouplingMap + Aer NoiseModel.

The JSON files in processors/ must be the OFFICIAL definitions from the
Challenge Kit. The noise block is read flexibly (several key aliases) because
the kit's exact key names are not known in advance; a clear error is raised
if no numbers can be found.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

REQUIRED_KEYS = ["name", "num_qubits", "coupling_map", "basis_gates"]

_ALIASES = {
    "one_qubit_error": ["one_qubit_error", "single_qubit_error", "p1", "error_1q", "e1q", "1q_error"],
    "two_qubit_error": ["two_qubit_error", "cx_error", "p2", "error_2q", "e2q", "2q_error"],
    "readout_error": ["readout_error", "measurement_error", "readout", "p_readout", "ro_error"],
}


def load_processor(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        proc = json.load(f)
    validate_processor(proc)
    return proc


def validate_processor(proc: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in proc]
    if missing:
        raise ValueError(f"Processor definition is missing keys: {missing}")
    n = int(proc["num_qubits"])
    for a, b in proc["coupling_map"]:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"Bad coupling edge {[a, b]} for {n} qubits")
    g = coupling_graph(proc)
    if g.number_of_nodes() != n or not nx.is_connected(g):
        raise ValueError(f"{proc['name']}: coupling graph must be connected and cover all {n} qubits")


def coupling_graph(proc: dict) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(proc["num_qubits"])))
    g.add_edges_from([tuple(e) for e in proc["coupling_map"]])
    return g


def build_coupling_map(proc: dict):
    """Bidirectional CouplingMap (direction is not part of this benchmark)."""
    from qiskit.transpiler import CouplingMap

    edges = []
    for a, b in proc["coupling_map"]:
        edges.append((int(a), int(b)))
        edges.append((int(b), int(a)))
    return CouplingMap(edges)


def noise_parameters(proc: dict) -> dict:
    """Return {'one_qubit_error','two_qubit_error','readout_error'} as floats."""
    noise = proc.get("noise", {}) or {}
    out = {}
    for canon, names in _ALIASES.items():
        for nm in names:
            if nm in noise:
                out[canon] = float(noise[nm])
                break
    missing = [k for k in _ALIASES if k not in out]
    if missing:
        raise ValueError(
            f"{proc['name']}: could not find noise values {missing} in the 'noise' block "
            f"(found keys: {list(noise.keys())}). Copy the official numbers into the JSON "
            f"using the keys one_qubit_error / two_qubit_error / readout_error."
        )
    return out


def build_noise_model(proc: dict, override: dict | None = None):
    """Depolarizing 1q (sx,x) + 2q (cx) errors and symmetric readout error.

    `override` can replace any of the three numbers (used for the equal-noise control).
    rz is treated as a virtual, error-free gate.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    params = noise_parameters(proc)
    if override:
        params.update(override)
    p1, p2, pr = params["one_qubit_error"], params["two_qubit_error"], params["readout_error"]

    nm = NoiseModel(basis_gates=list(proc["basis_gates"]))
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), [g for g in ("sx", "x") if g in proc["basis_gates"]])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), [g for g in ("cx", "cz", "ecr") if g in proc["basis_gates"]])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - pr, pr], [pr, 1 - pr]]))
    return nm


def processor_summary(proc: dict) -> dict:
    g = coupling_graph(proc)
    deg = [d for _, d in g.degree()]
    return {
        "name": proc["name"],
        "num_qubits": int(proc["num_qubits"]),
        "num_edges": g.number_of_edges(),
        "max_degree": max(deg),
        "avg_degree": round(sum(deg) / len(deg), 3),
        "diameter": nx.diameter(g),
        "avg_shortest_path": round(nx.average_shortest_path_length(g), 3),
        "basis_gates": ",".join(proc["basis_gates"]),
        **{k: v for k, v in noise_parameters(proc).items()},
        "placeholder": bool(proc.get("placeholder", False)),
    }
