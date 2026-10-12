# Qiskit Fall Fest 2026 - Phase 1 - O3 Knapsack

**Selected Problem Statement:** `O3 - Knapsack` (Track: Quantum Optimization)

Goal: solve a 0/1 knapsack instance with QAOA, run the *same* solution on **Processor A** and
**Processor B**, and explain how the two architectures change the compiled circuit and the result,
using measured evidence.

## Problem instance

Official O3 instance (from the Challenge material, `Problem_Statment_Details.pdf`): capacity 15,
values [10, 4, 7, 5, 8], weights [5, 3, 4, 2, 6] for items A to E.
Classical optimum (brute force over all 32 subsets): **A + B + C + D**, weight 14, value **26**.

## Processors (official coupling maps)

* Processor A: 5 qubits, edges (0,1), (1,2), (1,3), (3,4).
* Processor B: 9 qubits, edges (0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0).
* Gate set RZ, SX, X, CX on both. Noise: depolarizing 1q = 0.1 %, 2q = 0.5 %; readout error is not specified
  in the provided materials and is set to 0 (see `noise_note` in the processor JSON files).

## Method (short)

1. One qubit per item (5 qubits). Capacity handled by *unbalanced penalization* (no slack qubits):
   `cost(x) = -value(x) + lam1*(C-W) + lam2*(C-W)^2`, converted to an Ising Hamiltonian.
2. QAOA (main benchmark p = 1, depth sweep p = 1..3). Parameters tuned offline on an exact numpy simulator, then reused unchanged on A and B.
3. For each processor: transpile (5 transpiler seeds), run ideal and noisy on Qiskit Aer, record
   problem metrics (P(optimal), P(feasible), approximation ratio, Hellinger fidelity) and architecture metrics
   (depth, two-qubit gates, SWAPs, physical qubits, runtime).
4. Equal-noise control and a QAOA depth sweep separate connectivity effects from noise effects.

## Repository structure

```
README.md            this file
main.ipynb           end-to-end reproducible experiment (run top to bottom)
requirements.txt     dependencies
data/raw/            knapsack_instance.json
data/processed/      classical baseline, Ising encoding, tuned QAOA parameters
processors/          processor_A.json, processor_B.json  (official Challenge Kit definitions)
src/                 problem.py, processors.py, routing.py, metrics.py, visualization.py
results/tables/      CSV/JSON result tables
results/figures/     graphs, circuits, comparison plots
report.md            analysis (auto-filled with measured numbers by the notebook)
assets/              equation images
```

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```

or open `main.ipynb` (Colab / Jupyter) and choose *Run all*. Run it from the repository root.
Everything is seeded (transpiler seeds 11-15, simulator seeds, optimizer seed 7).

## Key results

<!-- RESULTS:START -->
**Problem:** 5 items, capacity 15; exact optimum A+B+C+D (value 26). QAOA p=1, 4096 shots, 5 transpiler seeds.

| Metric | Processor A | Processor B | Difference (B - A) |
|---|---|---|---|
| Problem metric: P(optimal), noisy | 0.203 +/- 0.001 | 0.209 +/- 0.004 | 0.006 |
| P(feasible), noisy | 0.854 +/- 0.006 | 0.856 +/- 0.004 | 0.002 |
| Best sampled value / optimum, noisy | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 |
| Approx. ratio (mean feasible value / optimum) | 0.631 +/- 0.005 | 0.635 +/- 0.006 | 0.003 |
| Depth | 60.6 +/- 1.2 | 51.0 +/- 4.1 | -9.6 |
| 2Q gates (CX) | 32.6 +/- 1.2 | 26.0 +/- 0.0 | -6.6 |
| SWAPs | 4.2 +/- 0.4 | 2.0 +/- 0.0 | -2.2 |
| Physical qubits used | 5.0 +/- 0.0 | 5.0 +/- 0.0 | 0.0 |
| Runtime: transpile (s) | 0.0166 +/- 0.0063 | 0.0237 +/- 0.0008 | 0.0072 |
| Runtime: noisy simulation (s) | 0.0345 +/- 0.0056 | 0.0364 +/- 0.0039 | 0.0018 |
| Fidelity: Hellinger vs ideal, noisy | 0.984 +/- 0.002 | 0.988 +/- 0.002 | 0.004 |

<!-- RESULTS:END -->

See `report.md` for the full interpretation and limitations.

## AI assistance disclosure

Parts of this repository were produced with AI assistance (Claude). The team reviewed, ran and takes
responsibility for the submitted work.
