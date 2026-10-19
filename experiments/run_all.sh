#!/bin/bash
# All simulation results of the letter, sequentially on one core (about 9 h): the aluminium-girder series
# (missions, post-lift tests), the contact tests of Section V-F, the mass sweep beyond the rated payload, the
# compiled-model checks, and the evaluations.
set -e
cd "$(dirname "$0")/.."
./experiments/run_aluminium.sh
./experiments/run_contact_tests.sh > logs/run_contact_tests.out 2>&1
./experiments/run_sweep.sh
./experiments/run_exact_all.sh > logs/exact_all.log 2>&1
python experiments/mission_metrics.py > logs/metrics.log 2>&1
python experiments/model_error_offset.py > logs/model_error_offset.log 2>&1
python experiments/model_error_offset.py kz0 > logs/model_error_offset_kz0.log 2>&1
python experiments/extract_traj.py al_n4_kz0 > logs/extract_traj.log 2>&1
echo ALL DONE
