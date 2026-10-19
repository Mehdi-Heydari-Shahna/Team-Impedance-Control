#!/bin/bash
# MuJoCo results of the letter for the 2.4 m aluminium girder with the HEA 280 cross-section (63 kg; the default
# profile "AL 280"), run sequentially on one core (about 5 h). The tests of Section V-F are in run_contact_tests.sh,
# the mass sweep beyond the rated payload in run_sweep.sh, the compiled-model checks in run_exact_all.sh; run_all.sh
# runs all of them.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R="python experiments/mission_run.py"
S=results/mission/state_lift_al_n4_kz0.pkl

# main run (N = 4, K_z = 0); stop if it does not complete
$R --n 4 --tag al_n4_kz0 --kz 0 > logs/mission_al_n4_kz0.log 2>&1
python -c "import numpy as np, sys; sys.exit(int(np.load('results/mission/al_n4_kz0.npz')['failed'][0] >= 0))"

# post-lift state and the tests from it (Section V-E, Fig. 7(a); timing of Section V-G)
python experiments/save_state.py --n 4 --kz 0 > logs/save_state_al.log 2>&1
python experiments/mujoco_push.py $S 100 > logs/al_push_100.log 2>&1
python experiments/mujoco_push.py $S 100 nofriction > logs/al_push_100_nofriction.log 2>&1
python experiments/mujoco_bias_sweep.py $S 63.07,58,53,48,43,38 > logs/al_bias_sweep.log 2>&1
python experiments/timing.py $S 3 > logs/al_timing.log 2>&1

# capacity estimate at 63 kg
python experiments/capacity.py 63.07 4,3,2 > logs/al_capacity.log 2>&1

# internal-channel variants, lambda_d = 0, team size
$R --n 4 --tag al_n4 --kz 1 > logs/mission_al_n4.log 2>&1 || true
$R --n 4 --tag al_n4_ff --internal feedforward > logs/mission_al_n4_ff.log 2>&1 || true
$R --n 4 --tag al_n4_noshare_kz0 --no-share-internal --kz 0 > logs/mission_al_n4_noshare_kz0.log 2>&1 || true
$R --n 3 --tag al_n3_kz0 --kz 0 > logs/mission_al_n3_kz0.log 2>&1 || true
$R --n 2 --tag al_n2_kz0 --kz 0 > logs/mission_al_n2_kz0.log 2>&1 || true

# force/torque-sensor noise, kinematic model errors (Fig. 4(c))
$R --n 4 --tag al_n4_noise5 --kz 0 --noise-f 5 --noise-m 0.5 --noise-seed 1 > logs/mission_al_n4_noise5.log 2>&1 || true
$R --n 4 --tag al_n4_noise15 --kz 0 --noise-f 15 --noise-m 1.5 --noise-seed 2 > logs/mission_al_n4_noise15.log 2>&1 || true
$R --n 4 --tag al_n4_kin_kz0 --kz 0 --kin-err 0.001 --kin-err-seed 3 > logs/mission_al_n4_kin_kz0.log 2>&1 || true
$R --n 4 --tag al_n4_kin --kz 1 --kin-err 0.001 --kin-err-seed 3 > logs/mission_al_n4_kin.log 2>&1 || true
$R --n 4 --tag al_n4_kin_ff --internal feedforward --kin-err 0.001 --kin-err-seed 3 > logs/mission_al_n4_kin_ff.log 2>&1 || true

# girder model of the controller (true 63.07 kg): Remark 3 bias with K_z = I (Fig. 7(d)), outcomes with K_z = 0,
# and psi = lambda_d with 70 kg
for m in 70 66 60 55 50 40; do
  $R --n 4 --tag al_n4_m${m} --kz 1 --ctrl-mass $m > logs/mission_al_n4_m${m}.log 2>&1 || true
done
for m in 70 66 60 55 50 40 30; do
  $R --n 4 --tag al_n4_m${m}_kz0 --kz 0 --ctrl-mass $m > logs/mission_al_n4_m${m}_kz0.log 2>&1 || true
done
$R --n 4 --tag al_n4_m70_ff --internal feedforward --ctrl-mass 70 > logs/mission_al_n4_m70_ff.log 2>&1 || true
echo ALUMINIUM DONE
