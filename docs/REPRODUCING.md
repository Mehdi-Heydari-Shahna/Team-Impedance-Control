# Reproduction workflow

`tools/reproduce.py` invokes the current aluminium-study scripts with the active Python interpreter. It runs sequentially with one BLAS/OpenMP thread and writes an isolated working copy outside the repository. The default directory is the sibling `.reproduction_work`; specify `--work-dir` to choose another location.

| Task | Work performed |
|---|---|
| `main` | main four-robot aluminium mission `al_n4_kz0` and completion check |
| `exact` | six exact-model experiments and four hold-period experiments |
| `figures` | exact-model and contact figures from supplied compact data |
| `capacity` | capacity grid listed in the original README, including 63.07 kg |
| `video` | main mission, completion check, two comparison missions, rendering and assembly |
| `full` | 72 jobs: all 37 missions, contact/exact checks, analyses, figures and video |

The full plan follows `experiments/run_all.sh`: `run_aluminium.sh`, `run_contact_tests.sh`, `run_sweep.sh`, `run_exact_all.sh`, mission metrics, model-error analysis with both gain settings, and trajectory extraction. It then executes the original README's teaser, snapshot, figure and video commands. The separate `capacity` task regenerates the full stored mass grid; `run_all.sh` itself only recomputes capacity at 63.07 kg, merging it with the supplied grid.

The main study uses the 63.0665 kg aluminium girder. The 183.36 kg and heavier masses in `run_sweep.sh` are intentional stress/capacity comparisons. Controller model-mass variants, contact amplitudes and random seeds follow the current scripts.

Append `--dry-run` to inspect a plan without creating a working copy or executing commands. Jobs with complete recorded outputs are reused on subsequent runs. Append `--force` to rerun jobs. Each job has a log under the working copy's `logs/` directory. The main mission must complete, matching the explicit check in `run_aluminium.sh`.

The runner copies a newly rendered `video/team_impedance_mission.mp4` back to this repository so that the publishing helper can upload it. It does not replace the supplied compact results. A full rerun's results remain in the working copy for comparison. The supplied release MP4 can be published directly without rerunning missions.

The video includes a designated no-internal-command comparison that loses the girder. A recorded mission failure in that comparison is an expected study outcome. A Python exception, missing output, nonfinite state or failed FFmpeg command is an execution failure and stops the workflow.

The numerical controller, dynamics, mission and supplied result bytes are preserved from the updated archive. Portability changes select the graphics backend by platform, locate fonts, create figure directories and normalize FFmpeg paths. Two video captions now distinguish exact-model/fixed-support certificates from empirical contact behavior. Large logs are local reproduction inputs, not Git data.
