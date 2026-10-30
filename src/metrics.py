import math
import numpy as np

from .problem import (
    KnapsackInstance,
    decode_bitstring,
    evaluate_bits,
    penalty_cost,
    bits_to_items,
)
def std_error(p: float, n: int) -> float:
    """Binomial standard error sqrt(p(1-p)/N)."""
    if n <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)



def solution_metrics(counts, inst, classical, enc, ideal_probs):
    """Calculate solution quality metrics from Qiskit measurement counts."""
    shots = sum(int(c) for c in counts.values())
    if shots <= 0:
        raise ValueError("Counts must contain at least one measured shot.")

    opt_value = int(classical["value"])
    feasible_count = 0
    p_opt_count = 0
    cost_sum = 0.0
    feasible_value_sum = 0.0
    best = None

    for key, count in counts.items():
        count = int(count)
        if count <= 0:
            continue

        # Qiskit displays classical bit 8 on the left.
        # Convert to item-first little-endian order.
        bit_string = key.replace(" ", "")
        bits = [int(b) for b in bit_string[::-1]]

        # Keep only the five item bits for knapsack evaluation.
        item_bits = bits[:len(inst.weights)]

        weight, value, feasible = evaluate_bits(inst, item_bits)

        cost_sum += count * penalty_cost(
            inst, bits, enc["lam1"], enc["lam2"]
        )

        if feasible:
            feasible_count += count
            feasible_value_sum += count * value

            if best is None or value > best[0]:
                best = (value, weight, item_bits)

            if value == opt_value:
                p_opt_count += count

    p_opt = p_opt_count / shots
    p_feas = feasible_count / shots
    best_value = best[0] if best is not None else 0
    mean_feasible = (
        feasible_value_sum / feasible_count
        if feasible_count else 0.0
    )

    return {
        "shots": shots,
        "prob_optimal": p_opt,
        "prob_optimal_se": std_error(p_opt, shots),
        "prob_feasible": p_feas,
        "prob_feasible_se": std_error(p_feas, shots),
        "best_feasible_value": int(best_value),
        "best_feasible_items": (
            bits_to_items(inst, best[2]) if best is not None else "(none)"
        ),
        "best_found_is_optimal": bool(best is not None and best_value == opt_value),
        "approx_ratio_best": best_value / opt_value,
        "mean_value_feasible": mean_feasible,
        "approx_ratio_mean_feasible": mean_feasible / opt_value,
        "mean_penalized_cost": cost_sum / shots,
        "hellinger_fidelity_vs_ideal": hellinger_fidelity(
            counts, ideal_probs, len(bit_string)
        ),
    }

def circuit_metrics(circuit):
    """Extract basic circuit resource metrics."""
    from collections import Counter

    counts = Counter(
        instruction.operation.name
        for instruction in circuit.data
    )

    return {
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
        "gate_counts": dict(counts),
        "cx_count": counts.get("cx", 0),
        "swap_count": counts.get("swap", 0),
        "two_qubit_gate_count": sum(
            count for gate, count in counts.items()
            if circuit.find_bit(
                next(
                    inst.qubits[0]
                    for inst in circuit.data
                    if inst.operation.name == gate
                )
            ) is not None
            and len(next(
                inst.qubits
                for inst in circuit.data
                if inst.operation.name == gate
            )) == 2
        ) if circuit.data else 0,
    }


def circuit_metrics(circuit):
    """Return basic Qiskit circuit resource metrics."""
    from collections import Counter

    counts = Counter(
        instruction.operation.name
        for instruction in circuit.data
    )

    two_qubit = sum(
        1 for instruction in circuit.data
        if len(instruction.qubits) == 2
    )

    return {
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
        "gate_counts": dict(counts),
        "cx_count": counts.get("cx", 0),
        "swap_count": counts.get("swap", 0),
        "two_qubit_gate_count": two_qubit
    }


def hellinger_fidelity(counts, ideal_probs, n):
    """Calculate Hellinger fidelity between measured and ideal distributions."""
    shots = sum(int(v) for v in counts.values())
    if shots <= 0:
        return 0.0

    measured = [0.0] * (2 ** n)

    for key, count in counts.items():
        bit_string = key.replace(" ", "")
        index = int(bit_string, 2)
        measured[index] += int(count) / shots

    ideal = list(ideal_probs)
    if len(ideal) != 2 ** n:
        raise ValueError("Ideal probability vector has incorrect length.")

    return float(
        sum((a ** 0.5) * (b ** 0.5)
            for a, b in zip(measured, ideal)) ** 2
    )
