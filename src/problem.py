
"""Phase 2 PS-O3 knapsack problem, slack-variable QUBO, Ising encoding,
and an exact NumPy QAOA reference.

Official Phase 2 formulation:
    min  -sum(values[i] * x[i])
         + P * (sum(weights[i] * x[i]) + s - capacity)**2

where:
    x[i] in {0, 1}
    s = s0 + 2*s1 + 4*s2 + 8*s3
    P = 5

Qubit layout:
    q0-q4: item variables A-E
    q5-q8: slack variables s0-s3

Bit-order convention:
    Qubit 0 is the RIGHTMOST character in a Qiskit count bitstring.
    Integer state-vector index bit i represents qubit i.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json

import numpy as np


PENALTY = 5.0
SLACK_WEIGHTS = [1, 2, 4, 8]
N_ITEM_QUBITS = 5
N_SLACK_QUBITS = 4
N_QUBITS = N_ITEM_QUBITS + N_SLACK_QUBITS


@dataclass
class KnapsackInstance:
    name: str
    names: list
    weights: list
    values: list
    capacity: int


@dataclass
class Encoding:
    """Ising representation H = constant + sum(h_i Z_i) + sum(J_ij Z_i Z_j)."""

    n: int
    constant: float
    h: dict
    J: dict
    diagonal: np.ndarray
    qubo_linear: np.ndarray
    qubo_quadratic: np.ndarray


def load_instance(path):
    """Load an instance JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    instance = KnapsackInstance(
        name=data.get("name", data.get("problem", "PS-O3")),
        names=list(data.get("names", data.get("items", ["A", "B", "C", "D", "E"]))),
        weights=list(data["weights"]),
        values=list(data["values"]),
        capacity=int(data["capacity"]),
    )

    if not (
        len(instance.names) == len(instance.weights) == len(instance.values) == 5
    ):
        raise ValueError("PS-O3 must contain exactly five items: A, B, C, D, E.")

    if any(weight <= 0 for weight in instance.weights):
        raise ValueError("Item weights must be positive.")

    if instance.capacity < 0:
        raise ValueError("Capacity cannot be negative.")

    return instance


def evaluate_bits(inst, bits):
    """Return (total_weight, total_value, feasible) for the five item bits."""
    bits = np.asarray(bits, dtype=int).reshape(-1)

    if len(bits) < len(inst.weights):
        raise ValueError("At least five item bits are required.")

    item_bits = bits[:len(inst.weights)]

    if not np.all(np.isin(item_bits, [0, 1])):
        raise ValueError("Bits must contain only 0 or 1.")

    weight = int(np.dot(inst.weights, item_bits))
    value = int(np.dot(inst.values, item_bits))
    return weight, value, weight <= inst.capacity


def decode_bitstring(bitstr):
    """Reverse a Qiskit count string so index 0 corresponds to qubit 0."""
    return bitstr.replace(" ", "")[::-1]


def bits_to_items(inst, bits):
    """Convert the first five bits into selected item names."""
    bits = np.asarray(bits, dtype=int).reshape(-1)
    return [
        inst.names[i]
        for i in range(len(inst.names))
        if bits[i] == 1
    ]


def brute_force(inst):
    """Find the exact classical optimum by checking all 2^5 item selections."""
    best = None
    records = []

    for mask in range(1 << len(inst.weights)):
        bits = np.array(
            [(mask >> i) & 1 for i in range(len(inst.weights))],
            dtype=int,
        )
        weight, value, feasible = evaluate_bits(inst, bits)

        record = {
            "bits": bits.tolist(),
            "items": bits_to_items(inst, bits),
            "weight": weight,
            "value": value,
            "feasible": feasible,
        }
        records.append(record)

        if feasible and (best is None or value > best["value"]):
            best = record

    return best, records


def _qubo_coefficients(inst, penalty=PENALTY):
    """Build QUBO coefficients for the five item and four slack variables.

    Cost is represented as:
        constant + sum(a[i] * y[i]) + sum(b[i,j] * y[i] * y[j])

    Here y = [xA, xB, xC, xD, xE, s0, s1, s2, s3].
    """
    n_items = len(inst.weights)

    if n_items != N_ITEM_QUBITS:
        raise ValueError("This Phase 2 encoding expects exactly five items.")

    coefficients = np.array(
        list(inst.weights) + SLACK_WEIGHTS,
        dtype=float,
    )
    n = len(coefficients)

    # Expand P * (sum(coefficients[i] * y[i]) - capacity)^2.
    # Binary variables satisfy y[i]^2 = y[i].
    linear = penalty * (
        coefficients**2 - 2.0 * inst.capacity * coefficients
    )
    linear[:n_items] -= np.asarray(inst.values, dtype=float)

    quadratic = np.zeros((n, n), dtype=float)

    for i, j in combinations(range(n), 2):
        quadratic[i, j] = (
            2.0 * penalty * coefficients[i] * coefficients[j]
        )

    constant = penalty * float(inst.capacity**2)
    return constant, linear, quadratic


def penalty_cost(inst, bits, lam1=None, lam2=PENALTY):
    """Evaluate the Phase 2 QUBO cost for a nine-bit item-plus-slack assignment.

    lam2 is retained as a compatibility parameter and represents P.
    lam1 is unused because the official Phase 2 formulation uses slack bits,
    not the old unbalanced linear penalty.
    """
    del lam1

    bits = np.asarray(bits, dtype=int).reshape(-1)

    if len(bits) != N_QUBITS:
        raise ValueError(
            "Phase 2 penalty_cost requires 9 bits: 5 item bits + 4 slack bits."
        )

    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("Bits must contain only 0 or 1.")

    item_weight = float(np.dot(inst.weights, bits[:N_ITEM_QUBITS]))
    item_value = float(np.dot(inst.values, bits[:N_ITEM_QUBITS]))
    slack_value = float(np.dot(SLACK_WEIGHTS, bits[N_ITEM_QUBITS:]))

    return (
        -item_value
        + float(lam2) * (item_weight + slack_value - inst.capacity) ** 2
    )


def cost_diagonal(inst, lam1=0.0, lam2=PENALTY):
    """Return QUBO costs for all 512 computational basis states."""
    del lam1
    diagonal = np.empty(1 << N_QUBITS, dtype=float)

    for index in range(1 << N_QUBITS):
        bits = [(index >> i) & 1 for i in range(N_QUBITS)]
        diagonal[index] = penalty_cost(inst, bits, lam2=lam2)

    return diagonal


def ising_coefficients(inst, lam1=0.0, lam2=PENALTY, scale=1.0):
    """Convert the Phase 2 QUBO into Ising coefficients.

    Uses y[i] = (1 - Z[i]) / 2.
    The returned constant is the identity-term offset.
    """
    del lam1

    constant_qubo, linear_qubo, quadratic_qubo = _qubo_coefficients(
        inst, penalty=lam2
    )

    n = len(linear_qubo)
    constant = float(constant_qubo)
    h = {}
    J = {}

    # Convert linear QUBO terms a_i*y_i.
    for i in range(n):
        constant += linear_qubo[i] / 2.0
        h[i] = -linear_qubo[i] / 2.0

    # Convert quadratic terms b_ij*y_i*y_j.
    for i, j in combinations(range(n), 2):
        coefficient = quadratic_qubo[i, j]
        if coefficient != 0:
            constant += coefficient / 4.0
            h[i] -= coefficient / 4.0
            h[j] -= coefficient / 4.0
            J[(i, j)] = coefficient / 4.0

    constant *= scale
    h = {i: value * scale for i, value in h.items()}
    J = {pair: value * scale for pair, value in J.items()}

    diagonal = cost_diagonal(inst, lam2=lam2) * scale

    return constant, h, J, diagonal


def ising_diagonal(inst, lam1=0.0, lam2=PENALTY, scale=1.0):
    """Return the diagonal cost array used by the exact NumPy simulator."""
    del lam1
    return cost_diagonal(inst, lam2=lam2) * scale


def make_encoding(inst, lam1=0.0, lam2=PENALTY, scale=1.0):
    """Create the nine-qubit Phase 2 encoding."""
    del lam1

    constant, h, J, diagonal = ising_coefficients(
        inst, lam2=lam2, scale=scale
    )
    _, linear, quadratic = _qubo_coefficients(inst, penalty=lam2)

    return Encoding(
        n=N_QUBITS,
        constant=constant,
        h=h,
        J=J,
        diagonal=diagonal,
        qubo_linear=linear * scale,
        qubo_quadratic=quadratic * scale,
    )


def _apply_rx_all(state, beta, n):
    """Apply the QAOA X-mixer exp(-i * beta * sum(X)) to every qubit."""
    state = np.asarray(state, dtype=complex).copy()
    c = np.cos(beta)
    s = -1j * np.sin(beta)

    for q in range(n):
        bit = 1 << q

        for base in range(1 << n):
            if base & bit:
                continue

            other = base | bit
            a = state[base]
            b = state[other]

            state[base] = c * a + s * b
            state[other] = s * a + c * b

    return state


def qaoa_state(enc, n=None, gammas=None, betas=None):
    """Simulate the QAOA state exactly using NumPy.

    For p=1, pass one gamma and one beta, for example:
        qaoa_state(enc, gammas=[0.4], betas=[0.7])
    """
    n = enc.n if n is None else int(n)

    if n != enc.n:
        raise ValueError(f"Encoding has {enc.n} qubits, not {n}.")

    gammas = [0.0] if gammas is None else list(gammas)
    betas = [0.0] if betas is None else list(betas)

    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length.")

    state = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * enc.diagonal)
        state = _apply_rx_all(state, float(beta), n)

    return state


def qaoa_probabilities(enc, n=None, gammas=None, betas=None):
    """Return computational-basis probabilities for the exact QAOA state."""
    state = qaoa_state(enc, n=n, gammas=gammas, betas=betas)
    return np.abs(state) ** 2


def optimize_qaoa(
    enc,
    n=None,
    p=1,
    restarts=5,
    seed=7,
    maxiter=300,
    objective="energy",
):
    """Optimize QAOA parameters with SciPy when available.

    Returns a dictionary containing the best parameters, objective value,
    and optimizer result. Uses p QAOA layers.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "optimize_qaoa requires SciPy. Install it with: pip install scipy"
        ) from exc

    n = enc.n if n is None else int(n)

    if n != enc.n:
        raise ValueError(f"Encoding has {enc.n} qubits, not {n}.")

    rng = np.random.default_rng(seed)
    best_result = None

    def score(params):
        gammas = params[:p]
        betas = params[p:]
        probabilities = qaoa_probabilities(
            enc, n=n, gammas=gammas, betas=betas
        )

        if objective == "prob_opt":
            # Probability of the minimum-energy state(s).
            min_energy = np.min(enc.diagonal)
            mask = np.isclose(enc.diagonal, min_energy)
            return -float(probabilities[mask].sum())

        # Expected QUBO energy.
        return float(np.dot(probabilities, enc.diagonal))

    for _ in range(max(1, int(restarts))):
        initial = rng.uniform(0.0, np.pi, size=2 * p)
        result = minimize(
            score,
            initial,
            method="COBYLA",
            bounds=[(0.0, np.pi)] * (2 * p),
            options={"maxiter": int(maxiter)},
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    params = best_result.x
    return {
        "gammas": params[:p].tolist(),
        "betas": params[p:].tolist(),
        "objective": float(best_result.fun),
        "optimizer_result": best_result,
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }


def build_qaoa_circuit(enc, n=None, gammas=None, betas=None, measure=True):
    """Build the corresponding Qiskit QAOA circuit.

    Uses the Ising cost terms in enc and an RX mixer. This builds the
    nine-qubit logical circuit; processor routing/transpilation is separate.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "build_qaoa_circuit requires Qiskit. Install project requirements."
        ) from exc

    n = enc.n if n is None else int(n)

    if n != enc.n:
        raise ValueError(f"Encoding has {enc.n} qubits, not {n}.")

    gammas = [0.0] if gammas is None else list(gammas)
    betas = [0.0] if betas is None else list(betas)

    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length.")

    circuit = QuantumCircuit(n, n if measure else 0)
    circuit.h(range(n))

    for gamma, beta in zip(gammas, betas):
        # Ising cost unitary; the constant offset is a global phase.
        for q, coefficient in enc.h.items():
            if coefficient != 0:
                circuit.rz(2.0 * float(gamma) * coefficient, q)

        for (i, j), coefficient in enc.J.items():
            if coefficient != 0:
                circuit.rzz(2.0 * float(gamma) * coefficient, i, j)

        # X-mixer unitary.
        for q in range(n):
            circuit.rx(2.0 * float(beta), q)

    if measure:
        circuit.measure(range(n), range(n))

    return circuit
