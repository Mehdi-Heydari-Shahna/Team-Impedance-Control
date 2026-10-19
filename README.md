# Team-Impedance-Control

**External and Internal Impedance for Robot Teams Sharing a Load: Exact Decoupling and Decay Certificates**

Mehdi Heydari Shahna

This repository contains the simulation code, recorded results, figure scripts, and supplementary video of the accompanying manuscript. Four Apollo humanoids lift, carry, and place a 2.4 m aluminium girder with a mass of 63 kg. The study also considers smaller teams, heavier loads, model errors, and sensor noise.

The controller combines an external impedance for the motion of the shared object with an internal impedance for the forces that distribute its load across the robots. The analysis establishes exact decoupling and explicit decay certificates under the stated model assumptions. The experiments evaluate the controller on the compiled model and with MuJoCo contact physics.

## Demonstration video

https://github.com/user-attachments/assets/f5356a71-2634-4e66-98bc-e3da226f19f8

Four Apollo humanoids approach, lift, carry, and place the 63 kg aluminium girder. The video also shows the comparison without the internal command (65 s; MuJoCo contact simulation, 500 Hz control, 1 kHz integration).

[Full-quality video](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control/releases/download/v1.0.0/team_impedance_mission.mp4) · [Source code archive](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control/releases/download/v1.0.0/source_code.zip) · [Recorded results archive](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control/releases/download/v1.0.0/compact_results.zip)

## Main capabilities

- Object-level external impedance for compliant motion under an environment wrench.
- Internal-wrench control and load allocation through the actuator efforts that do not drive admissible team motion.
- A fitted-reference inner loop for teams with redundant actuation and changing support modes.
- Exact-model checks of decoupling, decay rates, reference displacements, and grasp/support feasibility.
- MuJoCo missions with actuator limits, frictional contacts, payload-model errors, sensor noise, and capacity comparisons.

## Repository contents

| Path | Content |
|---|---|
| `teamimp/` | Compiled team model, two-loop controller, load-sharing QP, mission, MuJoCo and exact-model plants, scene and rendering helpers |
| `experiments/` | Mission runs, exact-model checks, contact tests, capacity estimates, analyses, and figure scripts |
| `results/exact/` | Recorded results of the compiled-model checks |
| `results/mission/` | Mission summaries, contact-test time series, model-error studies, and computation times |
| `results/capacity.json` | Static capacity estimates for the team-size and mass comparisons |
| `paper/figures/` | The first-page image of the manuscript |
| `video/make_video.py` | Rendering and assembly of the supplementary video from mission logs |
| `tools/reproduce.py` | Portable reproduction workflow |
| `tools/check_environment.py`, `tools/check_release.py` | Environment, model, and recorded-file checks |
| `docs/` | Result index, reproduction instructions, and validation scope |
| `environment.yml`, `requirements.txt` | Miniforge environment and pinned Python packages |

## Getting started

Use **Python 3.11** with Miniforge. From a Miniforge Prompt, open the repository folder and run:

```bat
conda env create -f environment.yml
conda activate team-impedance
python tools/check_release.py
python tools/check_environment.py --render
```

If the environment already exists, use `conda env update -n team-impedance -f environment.yml` instead of creating it. The environment installs MuJoCo 3.16.0, the numerical packages, and FFmpeg. Rendering defaults to GLFW on Windows and EGL on Linux; an existing `MUJOCO_GL` setting is respected.

## Run single examples

```bat
python tools/reproduce.py main
python tools/reproduce.py exact
python tools/reproduce.py capacity
python tools/reproduce.py figures
```

`main` runs the four-robot aluminium-girder mission and checks its completion. `exact` runs the compiled-model checks, `capacity` evaluates the capacity grid, and `figures` draws the exact-model and contact plots from the recorded compact data.

A single mission can also be run directly:

```bat
python experiments/mission_run.py --n 4 --tag al_n4_kz0 --kz 0
```

The default payload is the aluminium `AL 280` girder. Options include `--n 3` or `--n 2` for smaller teams, `--no-share-internal` for zero internal command, `--internal feedforward` for direct internal-wrench feedforward, `--ctrl-mass` for the payload mass used by the controller, and `--noise-f`, `--noise-m`, and `--kin-err` for sensing and kinematic errors. Direct experiment commands write into the repository's `results/` folder; the portable workflow below uses a separate working directory.

## Reproduce the paper

```bat
python tools/reproduce.py full --dry-run
python tools/reproduce.py full
python tools/reproduce.py video
```

`full --dry-run` lists the commands without running them. `full` runs the 37 mission variants, contact tests, exact-model checks, analyses, figures, and video. `video` runs only the three missions needed by the video renderer and assembles the MP4. Full reproduction can take several hours and requires several GB for logs and frames.

The workflow runs sequentially in an isolated sibling `.reproduction_work` directory and keeps the recorded compact results intact. Completed jobs are reused after an interruption. Use `--work-dir PATH` to choose another location or `--force` to rerun completed jobs. The original Linux commands are in `experiments/run_all.sh` and its component scripts.

| Paper item | Data or script | Reproduction |
|---|---|---|
| Table II and Fig. 3: compiled-model checks | `results/exact/*.npz`, `experiments/exact_validation.py` | `python tools/reproduce.py exact` |
| Table III and mission outcomes | `results/mission/metrics.json`, `experiments/mission_metrics.py` | Full workflow |
| Fig. 6: capacity estimates and mass sweep | `results/capacity.json`, `experiments/capacity.py`, `experiments/run_sweep.sh` | Capacity task and full workflow |
| Fig. 7: contact and model-error tests | `results/mission/mujoco_*.json`, corresponding NPZ files, `model_error_offset*.json` | Full workflow; figure task uses the recorded data |
| Figs. 1, 4, and 5: scene, mission, and trajectories | `experiments/render_teaser.py`, `render_stills.py`, `extract_traj.py`, `figures.py` | Full workflow |
| Supplementary video | `video/make_video.py` | `python tools/reproduce.py video` |

See [reproduction instructions](docs/REPRODUCING.md) and the [result index](docs/RESULTS.md) for details.

## Recorded results

The repository includes 41 compact result files and summaries of 37 missions. The main run is `al_n4_kz0`, with a payload mass of 63.0665 kg. The 183.36 kg runs and heavier mass sweeps are capacity comparisons, beyond the nominal aluminium payload.

The recorded main mission completes in 91.03 s, with a maximum object-position error of 6.99 mm and a peak actuator utilization of 0.87. In the 25 N palm-load test, the maximum object displacement is 0.20 mm for the internal command and 1.94 mm for the hand-reference comparison. Computation times are recorded in `results/mission/timing.json` and depend on the hardware.

The multi-GB mission logs, trajectory file, and saved post-lift state are generated by the reproduction workflow. They are required to regenerate mission snapshots, trajectory plots, and the video. The compact result archives and the finished video are available in [release v1.0.0](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control/releases/tag/v1.0.0).

## Scope

The decoupling and decay certificates concern the ideal constrained model in a regular fixed mode, under the manuscript's local assumptions, with an exact model and a measured environment wrench where required. The contact simulations also include support switching, friction, sampling, and actuator limits; they provide numerical evaluations rather than a certificate for the complete hybrid mission.

The Apollo model is from MuJoCo Menagerie. These are simulation studies, not experiments on physical Apollo robots. Environment and validation details are given in [docs/VALIDATION.md](docs/VALIDATION.md).

## Paper and citation

Please cite the accompanying manuscript when using this work:

Mehdi Heydari Shahna. **External and Internal Impedance for Robot Teams Sharing a Load: Exact Decoupling and Decay Certificates.** Accompanying manuscript, 2026.

Citation metadata are provided in [CITATION.cff](CITATION.cff). The compiled-model formulation builds on [RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler).

## License

A project-wide license for the research code has not been specified. The Apollo assets retain their supplied Apache License 2.0; see [assets/apollo/LICENSE](assets/apollo/LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
