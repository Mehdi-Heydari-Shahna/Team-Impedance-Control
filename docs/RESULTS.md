# Result index

This release uses the updated aluminium study supplied in `team_impedance_code(2).zip`, accompanying the current manuscript. The main run is `al_n4_kz0`, with mass 63.06649681528663 kg. The code archive contains 41 compact result files and 37 mission summaries. Steel-mass and heavier runs are capacity-sweep comparisons intentionally included in this study.

| Manuscript material | Supplied data | Regeneration |
|---|---|---|
| Table II and Fig. 3: identities, decay, antagonism, reference displacements, feasibility | `results/exact/*.npz` | `python tools/reproduce.py exact` |
| Table III and mission text | `results/mission/metrics.json` | full rerun and metric extraction |
| Capacity estimates | `results/capacity.json` | `python tools/reproduce.py capacity` |
| Contact push, internal/reference commands, model-mass tests | `results/mission/mujoco_*.json`, corresponding NPZ time series, `model_error_offset.json`, `model_error_offset_kz0.json` | full rerun |
| Computation times | `results/mission/timing.json` | full rerun; machine dependent |
| Exact-model and contact plots | generated from compact data | `python tools/reproduce.py figures` |
| Mission/trajectory figures and rendered snapshots | generators supplied; raw logs omitted | full rerun |
| Supplementary video | MP4 provided as a release asset; two captions clarified | supplied video, or `python tools/reproduce.py video` |

## Stored numerical reference values

These values are checks of supplied files, not independent rerun claims.

- Main mission: completed at 91.03 s; maximum girder tracking deviation 6.988383 mm; internal-wrench error RMS 16.273505 N and median 8.338495 N; maximum actuator utilization 0.868793.
- 25 N palm-load command: maximum girder deviation 0.196812 mm; hand-reference comparison 1.940829 mm (`mujoco_refdisp_25.json`).
- Stored median control / QP times: 9.004086 / 3.435167 ms.

The manifest hashes every supplied compact-result file without changing its bytes. Some original JSON summaries use `Infinity` or `-Infinity` for infeasible/contact-margin cases. Python's JSON reader accepts this original encoding; strict JSON readers may reject it.

## Large reproduction inputs

The per-run `<tag>.npz` and `<tag>_logs.pkl`, trajectory NPZ and saved post-lift state were omitted from the code archive. The current video renderer reads `al_n4_kz0`, `al_n3_kz0` and `al_n4_noshare_kz0`. The portable runner regenerates these inputs outside this repository. The MP4 belongs in GitHub Releases; frames and raw logs are not committed.
