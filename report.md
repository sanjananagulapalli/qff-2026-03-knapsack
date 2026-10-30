# Phase 1 Report - O3 Knapsack

## 1. Problem

Selected Problem Statement: **O3 - Knapsack** (track: Quantum Optimization). Capacity = 15. Items: A (w=5, v=10); B (w=3, v=4); C (w=4, v=7); D (w=2, v=5); E (w=6, v=8).

Exact classical baseline (brute force over 32 subsets): optimum = **A+B+C+D**, weight 14, value **26** (1 optimal solution(s), 28 feasible subsets).

## 2. Method

* Encoding: one qubit per item (5 qubits), unbalanced penalization with no slack qubits: cost(x) = -value(x) + -3.0*(C-W) + 1.0*(C-W)^2, mapped to an Ising Hamiltonian (all-to-all ZZ couplings because every item has non-zero weight).
* Algorithm: QAOA with p = 1 layers. Parameters were tuned offline on an exact numpy simulator (objective: prob_opt) and reused unchanged on both processors. Ideal probability of the optimum = 0.230 (random guessing = 0.031).
* Controlled benchmark: same circuit, 4096 shots, transpiler optimization level 1, 5 transpiler seeds ([11, 12, 13, 14, 15]); values are mean over seeds (+/- std over seeds).
* Processor A noise assumption: 1q = 0.001, 2q = 0.005, readout = 0.0. Readout error is not specified in the provided materials, so it is set to 0. Update these three numbers if the official Challenge Kit freezes different values.
* Processor B noise assumption: 1q = 0.001, 2q = 0.005, readout = 0.0. Readout error is not specified in the provided materials, so it is set to 0. Update these three numbers if the official Challenge Kit freezes different values.
* Noise: depolarizing 1q/2q errors and readout error from each processor JSON; plus an equal-noise control (both processors given Processor A's error rates) to separate connectivity effects from noise effects.

## 3. Results (A vs B, official noise)

| Metric | Processor A | Processor B |
|---|---|---|
| Circuit depth | 60.6 +/- 1.2 | 51.0 +/- 4.1 |
| Two-qubit gates (CX) | 32.6 +/- 1.2 | 26.0 +/- 0.0 |
| SWAPs (routing stage) | 4.2 +/- 0.4 | 2.0 +/- 0.0 |
| Physical qubits used | 5.0 +/- 0.0 | 5.0 +/- 0.0 |
| P(optimal) ideal | 0.230 +/- 0.002 | 0.230 +/- 0.006 |
| P(optimal) noisy | 0.203 +/- 0.001 | 0.209 +/- 0.004 |
| P(feasible) noisy | 0.854 +/- 0.006 | 0.856 +/- 0.004 |
| Best sampled value / optimum (noisy) | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Approx. ratio of mean feasible (noisy) | 0.631 +/- 0.005 | 0.635 +/- 0.006 |
| Hellinger fidelity vs ideal (noisy) | 0.984 +/- 0.002 | 0.988 +/- 0.002 |
| Transpile time (s) | 0.0166 +/- 0.0063 | 0.0237 +/- 0.0008 |
| Noisy run time (s) | 0.0345 +/- 0.0056 | 0.0364 +/- 0.0039 |

Best feasible solution sampled (noisy): A -> A+B+C+D, B -> A+B+C+D (optimum = A+B+C+D).

## 4. Interpretation

**What changed?** Two-qubit gates: A = 32.6, B = 26; Processor B is better (difference 6.6).

**Routing.** SWAPs: A = 4.2, B = 2; Processor B is better (difference 2.2). The QAOA cost layer needs all 10 pairwise ZZ interactions per layer, so the logical interaction graph is complete (K5); a sparse coupling graph can only realise it by inserting SWAPs.

**Solution quality (official noise).** P(optimal): A = 0.203, B = 0.209; the difference (0.006) is within about 2 standard errors, so it is not statistically distinguishable.

**Equal-noise control (connectivity only).** P(optimal) with identical error rates: A = 0.203, B = 0.209; the difference (0.006) is within about 2 standard errors, so it is not statistically distinguishable. Two-qubit gates in this control: A = 32.6, B = 26.0.

**Could gate-set / noise differences also contribute?** Not in this benchmark: both processors use the same gate set (RZ, SX, X, CX) and the same error rates, so the equal-noise control is identical to the official run by construction. Any A-vs-B difference therefore comes from the coupling graph and the circuit the compiler produced for it. Noise still matters indirectly: every extra CX gate is one more noisy operation.

## 5. QAOA depth sweep (cost vs quality)

| p | ideal P(opt) | A: CX | A: noisy P(opt) | B: CX | B: noisy P(opt) |
|---|---|---|---|---|---|
| 1 | 0.230 | 32.0 | 0.203 | 26.0 | 0.210 |
| 2 | 0.434 | 70.0 | 0.326 | 53.3 | 0.350 |
| 3 | 0.456 | 110.3 | 0.305 | 84.7 | 0.336 |

## 6. Bottlenecks and Phase 2 readiness

* Routing overhead: the all-to-all reference circuit needs 20 two-qubit gates; Processor A needs 32.6 (+12.6) and Processor B needs 26.0 (+6.0).
* Main bottleneck: the knapsack cost layer is a complete interaction graph on 5 qubits; every missing coupling must be bridged with SWAPs, and each SWAP costs three CX gates that are exposed to noise.
* Phase 2 implication: a processor graph with a denser local neighbourhood (more cycles or higher degree among the qubits that are used) would cut routing overhead for this workload; extra qubits that are far from the used region do not help.
* Next experiments: a placement/routing study (initial layout choice) and a noise-aware variant that stops at a lower QAOA depth when the SWAP overhead makes deeper circuits unprofitable.

## 7. Limitations

* One small instance (5 items) on two finite processor models; this does not establish a universal architecture advantage.
* The Aer simulator with a simple depolarizing + readout noise model is not real hardware.
* Routing quality depends on the transpiler heuristic and seed; we report mean +/- std over seeds.
* QAOA parameters were tuned on an ideal simulator, not re-tuned under noise.
* SWAP counts come from a routing-only transpile with the same seed; the executed circuit may merge or cancel some SWAP-related CX gates.
* Statistical uncertainty is binomial (sqrt(p(1-p)/N)) per run plus the spread over transpiler seeds.

## 8. AI assistance disclosure

Parts of this repository were produced with AI assistance (Claude). The team reviewed, ran and takes responsibility for the submitted work.

## 9. Team analysis (our interpretation of the results)

**Where Processor B wins.** Processor B (9 qubits, 12 couplers, max degree 4) compiled the same QAOA circuit with about 26 two-qubit gates versus about 32.6 on Processor A (5 qubits, 4 couplers, a tree-like layout). That is roughly 20% fewer CX gates and about half the SWAPs (2 vs 4.2). The cost layer is a complete ZZ graph (K5), so every item interacts with every other item. A sparse 4-edge tree cannot host K5 without routing, and B's denser graph needs less of it.

**Why the final quality gap is small.** Both processors use the same gate set and the same error rates (0.1% on 1-qubit gates, 0.5% on CX). The noisy P(optimal) was about 0.203 on A and about 0.209 on B, against an ideal 0.230, and the difference is within roughly two standard errors. With so few two-qubit gates at 0.5% error, the extra SWAPs on A cost only a small amount of fidelity at this problem size. We do not claim B is better on solution quality. We claim B is better on routing cost, and that this advantage should grow as depth or error rates grow.

**Cost of routing.** The all-to-all reference needs about 20 CX (10 RZZ, 2 CX each). Routing adds about +12.6 CX on A and about +6 on B, so the processor topology, not the algorithm, decides the final circuit size.

**Both processors solved the problem.** Both found the true optimum A+B+C+D (weight 14, value 26), with a best-sampled-value to optimum ratio of 1.000. About 85% of shots were feasible. This is because the unbalanced-penalization encoding needs no slack qubits.

**Main bottleneck.** The K5 interaction graph forces SWAP insertion on every sparse device. Depth and SWAP count therefore scale with connectivity, not with the number of items.

**Phase 2 readiness.** Because the cost depends on the coupling map, the next steps are: a connectivity-aware qubit-to-item mapping, reducing the penalty terms to fewer interactions, trying a larger knapsack where B's connectivity advantage should show up in fidelity, and tuning QAOA parameters under noise instead of ideal-only.

**Caveats.** The readout error was not specified in the provided materials, so we assumed 0. Results are averaged over 5 transpiler seeds with 4096 shots each, and all parameters were tuned offline on an ideal simulator and reused unchanged on both processors.



# Phase 2 — Processor B vs Processor D

## 1. Experimental Setup

The Phase 2 experiment evaluates the 0/1 Knapsack optimization problem using
a nine-qubit QAOA encoding consisting of five item qubits and four slack qubits.

The problem has a capacity of 15 and five items:

| Item | Weight | Value |
|---|---:|---:|
| A | 5 | 10 |
| B | 3 | 4 |
| C | 4 | 7 |
| D | 2 | 5 |
| E | 6 | 8 |

The classical optimum is A, B, C and D, with total weight 14 and total value 26.

Both processors were evaluated using 4096 shots.

## 2. Noise Model

The Phase 2 simulations use the prescribed noise parameters:

- Single-qubit depolarizing error: 0.001
- Two-qubit CX depolarizing error: 0.005
- Symmetric readout bit-flip probability: 0.01

## 3. Noisy Simulation Results

| Metric | Processor B | Processor D |
|---|---:|---:|
| Shots | 4096 | 4096 |
| Optimal probability | 5.1758% | 5.0049% |
| Feasible probability | 87.4268% | 86.5479% |
| Best feasible value | 26 | 26 |
| Mean feasible value | 15.5956 | 15.5698 |
| Hellinger fidelity | 0.7522 | 0.7607 |

Both processors identified the classical optimal solution:
A, B, C and D, with value 26.

## 4. Discussion

Processor B achieved an optimal-solution probability of
5.18%, compared with 5.00% for Processor D.

Processor B also achieved a higher feasible-solution probability.
Processor D recorded a slightly higher Hellinger fidelity against the ideal
QAOA distribution.

These results indicate that the two topologies exhibit different performance
characteristics under the same simulation conditions. No single processor
dominates every metric.

## 5. Processor A Compatibility

Processor A contains five physical qubits, whereas the Phase 2 encoding
requires nine qubits. Therefore, Processor A could not be evaluated using
the same nine-qubit circuit and is reported as incompatible with this
experimental configuration. No performance result is claimed for Processor A.

## 6. Conclusion

The Phase 2 experiment successfully executed the nine-qubit QAOA circuit
on Processors B and D under the prescribed noise model. Both processors
recovered the optimal knapsack solution.

Processor B performed better in optimal and feasible sampling probabilities,
while Processor D showed slightly higher Hellinger fidelity. These findings
highlight the importance of evaluating multiple hardware-topology metrics
rather than relying on a single performance indicator.
