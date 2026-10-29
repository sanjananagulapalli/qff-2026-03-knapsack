
"""Processor A/B/D definitions -> Qiskit CouplingMap + Aer NoiseModel.

Processor configurations are loaded from the JSON files in processors/.
The noise model uses the official Phase 2 probabilities:
  - 1-qubit depolarizing error: 0.001 on rz, sx, and x
  - 2-qubit depolarizing error: 0.005 on cx
  - Symmetric readout bit-flip probability: 0.01
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

REQUIRED_KEYS = ["name", "num_qubits", "coupling_map", "basis_gates"]

_ALIASES = {
    "one_qubit_error": [
        "one_qubit_error", "single_qubit_error", "p1",
        "error_1q", "e1q", "1q_error"
    ],
    "two_qubit_error": [
        "two_qubit_error", "cx_error", "p2",
        "error_2q", "e2q", "2q_error"
    ],
    "readout_error": [
        "readout_error", "measurement_error", "readout",
        "p_readout", "ro_error"
    ],
}


def load_processor(path) -> dict:
    """Load and validate a processor JSON configuration."""
    with Path(path).open("r", encoding="utf-8") as f:
        proc = json.load(f)
    validate_processor(proc)
    return proc


def validate_processor(proc: dict) -> None:
    """Check required fields, valid edges, and connected topology."""
    missing = [k for k in REQUIRED_KEYS if k not in proc]
    if missing:
        raise ValueError(
            f"Processor definition is missing keys: {missing}"
        )

    n = int(proc["num_qubits"])

    if n < 1:
        raise ValueError("num_qubits must be positive")

    for edge in proc["coupling_map"]:
        if len(edge) != 2:
            raise ValueError(f"Invalid coupling edge: {edge}")

        a, b = map(int, edge)

        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(
                f"Bad coupling edge {[a, b]} for {n} qubits"
            )

    g = coupling_graph(proc)

    if g.number_of_nodes() != n or not nx.is_connected(g):
        raise ValueError(
            f"{proc['name']}: coupling graph must be connected "
            f"and cover all {n} qubits"
        )

    if not proc["basis_gates"]:
        raise ValueError("basis_gates must not be empty")


def coupling_graph(proc: dict) -> nx.Graph:
    """Create an undirected graph from the processor connections."""
    g = nx.Graph()
    g.add_nodes_from(range(int(proc["num_qubits"])))
    g.add_edges_from(
        [tuple(map(int, edge)) for edge in proc["coupling_map"]]
    )
    return g


def build_coupling_map(proc: dict):
    """Build a bidirectional Qiskit CouplingMap.

    The benchmark treats the supplied connections as bidirectional.
    """
    from qiskit.transpiler import CouplingMap

    edges = []
    for a, b in proc["coupling_map"]:
        a, b = int(a), int(b)
        edges.append((a, b))
        edges.append((b, a))

    return CouplingMap(edges)


def noise_parameters(proc: dict) -> dict:
    """Return the three official noise probabilities as floats."""
    noise = proc.get("noise", {}) or {}
    out = {}

    for canonical_name, aliases in _ALIASES.items():
        for name in aliases:
            if name in noise:
                out[canonical_name] = float(noise[name])
                break

    missing = [key for key in _ALIASES if key not in out]

    if missing:
        raise ValueError(
            f"{proc['name']}: missing noise values {missing}. "
            f"Found keys: {list(noise.keys())}. "
            "Provide one-qubit, two-qubit, and readout probabilities."
        )

    for name, probability in out.items():
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{proc['name']}: {name} must be between 0 and 1; "
                f"received {probability}"
            )

    return out


def build_noise_model(proc: dict, override: dict | None = None):
    """Build the official Phase 2 Aer noise model.

    Applies one-qubit depolarizing noise to rz, sx, and x;
    two-qubit depolarizing noise to cx; and symmetric readout error.

    `override` can replace any of the three probabilities for a
    controlled experiment. Normal benchmark runs should use JSON values.
    """
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
    )

    params = noise_parameters(proc)

    if override:
        params.update(override)

    for name in _ALIASES:
        if name not in params or not 0.0 <= float(params[name]) <= 1.0:
            raise ValueError(
                f"Invalid noise override for {name}: {params.get(name)}"
            )
        params[name] = float(params[name])

    p1 = params["one_qubit_error"]
    p2 = params["two_qubit_error"]
    pr = params["readout_error"]

    basis_gates = list(proc["basis_gates"])
    nm = NoiseModel(basis_gates=basis_gates)

    one_qubit_gates = [
        gate for gate in ("rz", "sx", "x")
        if gate in basis_gates
    ]
    two_qubit_gates = [
        gate for gate in ("cx",)
        if gate in basis_gates
    ]

    if one_qubit_gates:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1),
            one_qubit_gates,
        )

    if two_qubit_gates:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2),
            two_qubit_gates,
        )

    nm.add_all_qubit_readout_error(
        ReadoutError([
            [1.0 - pr, pr],
            [pr, 1.0 - pr],
        ])
    )

    return nm


def processor_summary(proc: dict) -> dict:
    """Calculate topology and noise statistics for reports and tables."""
    g = coupling_graph(proc)
    degrees = [degree for _, degree in g.degree()]

    return {
        "name": proc["name"],
        "num_qubits": int(proc["num_qubits"]),
        "num_edges": g.number_of_edges(),
        "max_degree": max(degrees),
        "avg_degree": round(sum(degrees) / len(degrees), 3),
        "diameter": nx.diameter(g),
        "avg_shortest_path": round(
            nx.average_shortest_path_length(g), 3
        ),
        "basis_gates": ",".join(proc["basis_gates"]),
        **noise_parameters(proc),
        "placeholder": bool(proc.get("placeholder", False)),
    }
