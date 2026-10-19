#!/bin/bash
# Contact tests of Section V-F (Fig. 7(b), (c)) from the post-lift state of the main run (written by
# run_aluminium.sh), with palm-load shifts scaled to the 63 kg aluminium girder (palm loads about 77 N):
# internal command versus reference displacements (z frozen) and the internal command with the internal impedance
# active, with and without the joint friction of the plant. Sequential, one core (about 50 min).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
S=results/mission/state_lift_al_n4_kz0.pkl
for a in 15 25 40; do
  python experiments/mujoco_refdisp.py $S $a > logs/al_refdisp_$a.log 2>&1
done
python experiments/mujoco_refdisp.py $S 25 nofriction > logs/al_refdisp_25_nofriction.log 2>&1
for a in 15 30 40 60; do
  python experiments/mujoco_decoupling.py $S $a > logs/al_decoupling_$a.log 2>&1
done
for a in 15 30; do
  python experiments/mujoco_decoupling.py $S $a nofriction > logs/al_decoupling_${a}_nofriction.log 2>&1
done
echo CONTACT TESTS DONE
