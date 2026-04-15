# Computing with Distance Geometry

Research code exploring the **Distance Geometry Problem (DGP)** as a computational framework, with an emphasis on reductions from **Binary Linear Programming (BLP)** to **DGP**, and on **upper/lower bound** techniques for a *minimum-error* DGP variant (MinErrDGP1).

This repository contains:

- Reductions: BLP → SAT (Warners-style CNF encoding) and multiple BLP → DGP constructions (`reductions/`).
- Solvers / bounds for MinErrDGP1 (Gurobi models + compiled heuristics / relaxations).
- Experiment/orchestration scripts that generate random instances, run solvers, and collect results (`*_orchestrator.py`, `solver_compare.py`).
- A short 2D DGP feasibility model (`DGP_2_solver.py`).

> For full methodology, definitions, and experimental discussion, see `Research_Report_15_06_2025.pdf`.

---

## Repository layout

**Top-level scripts**

- `DGP_2_solver.py` — parses a `.dat` DGP instance and solves a *2D feasibility* embedding with Gurobi (quadratic equalities).
- `solver_compare.py` — generates random BLP instances, reduces them to DGP, and compares solving approaches (Gurobi MinErrBLP vs. DGP-based pipeline).
- `k_clique_solver.py` — example pipeline on a k-clique BLP instance (`k_clique.opb`) reduced to DGP and solved with the compiled solver.
- `relaxation_orchestrator.py` — compares a relaxation lower bound (from `relax_gen2Dembedding.exe`) with an upper bound from `upper_bounds/minErrDGP1.py` on random instances.
- `cycle_relaxation_orchestrator.py` — large experiment runner comparing cycle-based and greedy-packing lower bounds; writes `experiment_results.csv`.
- `UB_orchestrator.py` / `UB_arbitrary_orchestrator.py` — scripts for benchmarking a “cheap” upper bound against a Gurobi upper bound, including on DGP instances produced by the BLP→DGP reduction.

**Key directories**

- `reductions/` — all problem reductions (BLP→SAT, BLP→DGP, optimized BLP→DGP, test harnesses).
- `solver/` — C++ branch-and-bound style solver (compiled binary invoked as `./solver/solver`). See `solver/Makefile` for build settings (Gurobi C++ API).
- `upper_bounds/` — Gurobi-based upper bounds for MinErrDGP1 (notably `upper_bounds/minErrDGP1.py`).
- `relaxation/` — compiled executables for relaxation-based lower bounds (cycle-based and greedy packing).

---

## Concepts (what the code is doing)

### Distance Geometry Problem (DGP)
Given a graph with weighted edges, the DGP asks for an embedding of vertices in Euclidean space such that the Euclidean distances match the edge weights. In this repo, instances are typically stored in an AMPL-style `.dat` format containing an edge list.

### MinErrDGP1
A 1D “minimum-error” DGP variant used throughout experiments. Instead of requiring exact distances, we minimize a sum of absolute edge-length errors (or related error measures), producing an **upper bound** objective value.

### Reductions (BLP → DGP)
The main research thrust is to encode feasibility/structure of a BLP instance into a DGP instance, so that geometric reasoning/solving can be applied to combinatorial problems. Implementations live in `reductions/blp2dgp.py` and `reductions/blp2dgp_opt.py`.

### BLP → SAT (Warners-style)
`reductions/blp2sat_linear_reduction.py` implements a linear-time transformation of linear inequalities into CNF (based on J.P. Warners’ method), used as a baseline/comparison reduction path.

---

## Getting started

### 1) Clone

```bash
git clone https://github.com/maelkupp/Computing-with-Distance-Geometry.git
cd Computing-with-Distance-Geometry
```

### 2) Python environment

Most scripts are plain Python. Several require **Gurobi** (`gurobipy`) and a working Gurobi license.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install numpy pandas gurobipy
```

> Note: the repository currently contains a `venv/` directory committed to source control. It is generally recommended to remove it and use a local `.venv/` instead.

---

## Input formats

### DGP `.dat` format
Many tools expect an AMPL-style edge list like:

```text
param : E : c I :=
  1 2  3.000 0
  2 3  1.000 0
;
```

Some writers in this repo also append vertex “labels” in a comment, e.g. `# [x1,x2]`.

### BLP `.opb` format
BLP instances are read from OPB files (Pseudo-Boolean format), e.g. `k_clique.opb`. The parsers in this repo interpret terms like `+3*x2` and constraints ending in `<= b;`, `>= b;`, or `= b;`.

---

## How to use the code

### A) Solve a 2D DGP feasibility instance with Gurobi

```bash
python DGP_2_solver.py path/to/instance.dat
```

The script will:
- parse the edge list
- build a non-convex quadratic model
- apply “gauge fixing” (fixing the first edge to remove rotation/translation symmetries)
- output a feasible embedding if found within the time limit

### B) Compute a MinErrDGP1 upper bound with Gurobi

```bash
python upper_bounds/minErrDGP1.py path/to/instance.dat
```

It prints the upper-bound value on stdout (and a more verbose embedding/objective log on stderr).

### C) Run reduction pipeline on k-clique example

```bash
python k_clique_solver.py
```

This reads `k_clique.opb`, reduces it to a DGP `.dat` file, then calls the compiled solver binary:

- expected binary: `./solver/solver`
- expected output: objective value + runtime (see `solver_compare.py::run_dgp_solver`)

### D) Compare solvers / reductions on random instances

```bash
python solver_compare.py 10 20 20 0.3 feas
# args: num_tests n_vars n_cons density type
# type ∈ {feas, rnd, infeas, perturb}
```

This script generates OPB instances, performs BLP→DGP reductions (both “opt” and baseline), writes `*-dgp.dat` files, and runs the DGP solver.

### E) Run relaxation experiments (lower bounds)

- `relaxation_orchestrator.py` calls `relax_gen2Dembedding.exe` and compares its relaxation value against a Gurobi UB computed on `temp_graph.dat`.
- `cycle_relaxation_orchestrator.py` runs many experiments and writes `experiment_results.csv`.

These scripts rely on compiled executables in `relaxation/` and (for some sizes) on `upper_bounds/minErrDGP1.py`.

---

## Building the C++ solver

The C++ solver in `solver/` uses the **Gurobi C++ API**. The Makefile currently assumes a Linux install path:

- `GUROBI_HOME = /opt/gurobi1100/linux64`

To build:

```bash
cd solver
make
```

If your Gurobi is installed elsewhere, edit `solver/Makefile` accordingly.

---

## Reproducing experiments

- `cycle_relaxation_orchestrator.py` sweeps over multiple `n` and density levels, runs two relaxation executables, and for small `n` also computes a Gurobi upper bound. Output is stored in `experiment_results.csv`.
- `UB_orchestrator.py` and `UB_arbitrary_orchestrator.py` run repeated trials and print per-trial ratios and timing comparisons between heuristics and Gurobi bounds.

---

## Notes / caveats

- Many scripts assume Windows-style executable names/paths (e.g. `relaxation\...exe`) while others use Unix-style (`./solver/solver`). You may need to adjust paths depending on OS.
- Several experiments require compiled binaries that are included in the repo (`*.exe`) but may not run on non-Windows platforms.
- Gurobi is a commercial solver; you need a valid license for `gurobipy` and for building the C++ solver.

---

## Acknowledgments

Research conducted under the supervision of Prof. Leo Liberti at École Polytechnique.
