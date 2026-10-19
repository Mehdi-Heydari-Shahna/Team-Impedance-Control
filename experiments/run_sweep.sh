#!/bin/bash
# Mass sweep with the same girder cross-section beyond the rated payload of the robots (K_z = 0): runs around the
# capacity estimate U* = 1 for N = 2, 3, 4, including 183 kg (the HEA 280 steel mass). Sequential, one core (~2 h).
cd "$(dirname "$0")/.."
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R="python experiments/mission_run.py --kz 0"
$R --n 4 --mass 183.36 --tag n4_kz0 > logs/mission_n4_kz0.log 2>&1
$R --n 3 --mass 183.36 --tag n3_kz0 > logs/mission_n3_kz0.log 2>&1
$R --n 2 --mass 183.36 --tag n2_kz0 > logs/mission_n2_kz0.log 2>&1
for nm in 4:240 4:260 4:280 3:120 3:140 3:160 2:100 2:120 2:140; do
  n=${nm%%:*}; m=${nm##*:}
  $R --n $n --mass $m --tag sweep_n${n}_${m} > logs/sweep_n${n}_${m}.log 2>&1
done
echo SWEEP DONE
