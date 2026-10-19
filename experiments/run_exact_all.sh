#!/bin/bash
# Exact-model validation (Table II, Fig. 3: Theorems 1-2, Proposition 1, Corollary 1); runs sequentially on one core.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for e in identities rates antagonism refdisp decoupling corollary; do
  python experiments/exact_validation.py $e
done
for ts in 4 2 1 0.5; do
  python experiments/exact_validation.py zoh $ts
done
