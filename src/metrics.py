import math
import numpy as np
def std_error(p: float, n: int) -> float:
    """Binomial standard error sqrt(p(1-p)/N)."""
    if n <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def solution_metrics(
    counts: dict,
    inst: KnapsackInstance,
    classical: dict,
    enc: dict,
    ideal_probs: np.ndarray,
) -> dict:
    """Calculate sampled solution quality, feasibility and distribution metrics."""
    shots = int(sum(counts.values()))
    if shots <= 0:
        raise ValueError("Counts must contain at least one measured shot.")

    opt_value = int(classical["best_value"])
    if opt_value <= 0:
        raise ValueError("Classical optimum must be positive.")

    p_opt_count = 0
    feasible_count = 0
    cost_sum = 0.0
    feasible_value_sum = 0.0
    best = None  # (value, weight, bits)

    for key, count in counts.items():
        count = int(count)
        if count < 0:
            raise ValueError("Measurement counts cannot be negative.")
        if count == 0:
            continue

        bits = decode_bitstring(key)
        weight, value, feasible = evaluate_bits(inst, bits)

        cost_sum += count * penalty_cost(
            inst, bits, enc["lam1"], enc["lam2"]
        )

        if feasible:
            feasible_count += count
            feasible_value_sum += count * value

            if best is None or value > best[0]:
                best = (value, weight, bits)

            if value == opt_value:
                p_opt_count += count

    p_opt = p_opt_count / shots
    p_feas = feasible_count / shots
    best_value = best[0] if best is not None else 0
    mean_feasible = (
        feasible_value_sum / feasible_count
        if feasible_count > 0 else 0.0
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
        "best_found_is_optimal": bool(
            best is not None and best_value == opt_value
        ),
        "approx_ratio_best": best_value / opt_value,
        "mean_value_feasible": mean_feasible,
        "approx_ratio_mean_feasible": mean_feasible / opt_value,
        "mean_penalized_cost": cost_sum / shots,
        "hellinger_fidelity_vs_ideal": hellinger_fidelity(
            counts, ideal_probs, inst.n
        ),
    }
