# Two-loop external/internal impedance for humanoid teams sharing a load (MuJoCo study)

Simulation code for Section V of the letter *External and Internal Impedance for Robot Teams Sharing a Load:
Exact Decoupling and Decay Certificates*. Four Apptronik Apollo humanoids (MuJoCo Menagerie model) carry a 2.4 m
aluminium I-beam with the cross-section of an HEA 280 girder (63 kg, about 16 kg per robot, within the 25 kg payload
published for Apollo) on a construction site: go, pick, lift, carry, place, unload, step back. A mass sweep with the
same cross-section (up to 280 kg, beyond the rated payload) tests the capacity estimate of the controller.

Numbering follows the letter: Fig. 1 first-page image, Fig. 3 compiled-model checks, Fig. 4 mission, Fig. 5
trajectories, Fig. 6 capacity, Fig. 7 contact tests; Table I parameters, Table II compiled-model checks, Table III
mission outcomes.

## Layout

| Path | Content |
|---|---|
| `teamimp/team.py` | compiled team model of a support/grasp mode: `A`, `E`, `B`, `L = B^T E`, grasp matrix `G`, `D_G`, `P_I`, `V_I`, effort split (Lemma 1), `Psi^+ = (Psi~ Pi_L)^+`, `N(L^T) ∩ N(Psi~)` |
| `teamimp/controller.py` | two-loop controller: fitted reference (3), required effort (4), inner loop (5), external impedance (6), required rates (7), lift (8), internal impedance (9) with conditional integration (anti-windup) while an actuator is at its limit |
| `teamimp/share.py` | load sharing through the internal channel: QP over `lambda_d` and the foot efforts with the margins of Corollary 1 (contact-wrench cones of palms and soles; palm margins weighted ten times the sole margins) and torque limits; capacity estimate `U*` |
| `teamimp/sim_exact.py`, `plant_exact.py` | exact compiled model (1) as the plant (RK4 with constraint projection), optional zero-order hold |
| `teamimp/sim_mj.py` | MuJoCo plant (soft frictional contacts, joint friction, torque limits), contact-wrench sensing with optional sensor noise, 500 Hz control; the rate-gain term `-K B^T v` of (5) acts continuously as actuator damping |
| `teamimp/mission.py` | mission phases (gait as a sequence of compiled support modes, pick, load transfer, lift, carry, place, unload, release); kinematic model errors of the controller |
| `teamimp/scene.py`, `textures.py`, `render.py` | construction-site scene, girder profiles (steel HEA 240-300 and the aluminium "AL 280"), trestles, rendering helpers |
| `experiments/run_all.sh` | all results of the letter, sequentially on one core (about 9 h): `run_aluminium.sh`, `run_contact_tests.sh`, `run_sweep.sh`, `run_exact_all.sh` and the evaluations |
| `experiments/run_aluminium.sh` | the aluminium-girder missions (variants, team size, noise, kinematic errors, girder-model errors), the post-lift state and the push, standstill-bias and timing tests |
| `experiments/run_contact_tests.sh` | the tests of Section V-F from the post-lift state (palm-load shifts of 15-60 N, with and without joint friction) |
| `experiments/run_sweep.sh` | mass sweep with the same cross-section beyond the rated payload (N = 2, 3, 4) |
| `experiments/exact_validation.py`, `run_exact_all.sh` | Table II, Fig. 3: Theorem 1 identities, decoupling (exact/inexact model), zero-order hold, Theorem 2 rates, Proposition 1(iii)-(iv), Corollary 1 |
| `experiments/capacity.py` | capacity estimate `U*` (Fig. 6(a), Table III) |
| `experiments/mission_run.py` | one MuJoCo mission run (team size, girder profile and mass, internal-channel variant, girder model of the controller, sensor noise, kinematic model errors) |
| `experiments/mission_metrics.py` | metrics of Table III and of the text of Section V (`results/mission/metrics.json`) |
| `experiments/extract_traj.py` | trajectories of the main run for Fig. 5 |
| `experiments/save_state.py` | runs the main mission to the end of the lift and stores the state used by the tests below |
| `experiments/mujoco_push.py` | Fig. 7(a): 100 N push, with and without joint friction |
| `experiments/mujoco_refdisp.py` | Fig. 7(b)-(c): internal command versus hand-reference, frozen and projected actuator-reference displacements (option `nofriction`) |
| `experiments/mujoco_decoupling.py` | internal command with the internal impedance active (Section V-F; option `nofriction`) |
| `experiments/model_error_offset.py` | Fig. 7(d), Table III: Remark 3 bias for girder-model errors (runs with K_z = I; option `kz0` for the runs with K_z = 0), predicted and measured in the weight shifts of the carry |
| `experiments/mujoco_bias_sweep.py` | the same bias at a standstill, where static joint friction holds back part of it |
| `experiments/timing.py` | computation time per control step and per QP (Section V-G) |
| `experiments/figures.py`, `render_stills.py`, `render_teaser.py` | Figs. 3-7, snapshots of Fig. 4, Fig. 1 |
| `video/make_video.py` | rendering and assembly of the video |

## Requirements

Python 3.10+, `mujoco` (3.x), `numpy`, `scipy`, `daqp`, `matplotlib`, `pillow`, `trimesh`, `fast_simplification`
(asset preparation only), `ffmpeg` (video). Rendering uses `MUJOCO_GL=egl`. All runs use one CPU core
(`OPENBLAS_NUM_THREADS=1`); a full mission (91 s simulated) takes 6-10 min.

## Reproduce

```bash
./experiments/run_all.sh                             # all results (about 9 h)
python experiments/capacity.py 60,63.07,80,100,120,140,160,180,183,183.36,200,220,240,260,280,300 4,3,2

# figures and video
python experiments/render_teaser.py                  # Fig. 1 (paper/figures/fig_teaser.png)
python experiments/render_stills.py                  # snapshots of Fig. 4 (paper/figures/snap_*.png)
python experiments/figures.py all                    # Figs. 3-7
python video/make_video.py all                       # video/team_impedance_mission.mp4
```

Single runs: `python experiments/mission_run.py --n 4 --tag al_n4_kz0 --kz 0` (main run; the default girder is the
63 kg aluminium "AL 280"), `--mass 183.36` for the HEA 280 steel mass, `--internal feedforward` for `psi = lambda_d`,
`--no-share-internal` for `lambda_d = 0`, `--ctrl-mass` for the girder model of the controller, `--noise-f/--noise-m`
for sensor noise and `--kin-err` for kinematic model errors.

`results/` holds the summary results behind Tables II-III and Section V (`capacity.json`, `exact/*.npz`,
`mission/metrics.json`, `mission/mujoco_*.json|npz`, `mission/model_error_offset.json` and `_kz0.json`, `mission/timing.json`).
The per-run logs (`results/mission/<tag>.npz`, `<tag>_logs.pkl`, several GB), the trajectories of Fig. 5
(`traj_al_n4_kz0.npz`) and the post-lift state (about 0.4 GB) are not included; the commands above write them, and
Figs. 4-5 and the video are made from them. Timings depend on the machine.

## Assets

`assets/apollo`: Apptronik Apollo model from MuJoCo Menagerie (Apache License 2.0, see `assets/apollo/LICENSE`);
the visual meshes were decimated with `teamimp/decimate.py` (source directory in the environment variable
`APOLLO_SRC`) to reduce rendering cost. The scene textures are generated procedurally by `teamimp/textures.py`.
