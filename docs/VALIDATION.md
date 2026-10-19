# Validation scope

The following release-preparation checks passed on Linux with Python 3.12 and the pinned package versions, including MuJoCo 3.16.0:

- Python syntax and preserved hashes of 41 original compact result files and 16 unchanged numerical modules.
- Aluminium-study reference values and the index of 37 supplied mission summaries.
- Dependency imports, FFmpeg/libx264 availability, two-robot aluminium model compilation, and compiled dynamics API compatibility.
- One controller/plant execution step covering 2 ms of simulated time, with finite state and control values.
- A 160 x 120 off-screen frame using EGL.
- Exact-model and contact figure regeneration from the current supplied compact data, through the portable runner in an isolated working directory.
- Fifteen offline publishing tests, including the default supplied-video workflow. External GitHub operations were mocked; no repository was created by these tests.

These checks do not establish full-mission completion, independently reproduce all stored results, validate every manuscript conclusion, or establish real-time execution. The target Miniforge environment specifies Python 3.11; native Windows execution and full numerical agreement with the stored contact results were not tested here. Full missions and full video regeneration were not run during this release preparation.

The code archive omits raw mission logs and the saved post-lift state. A finished supplementary MP4 was supplied separately and is included as a release asset, with two caption clarifications. Regenerating the full experiments or the video requires running the reproduction commands.

The uploaded research environment did not pin a MuJoCo version. This release pins MuJoCo 3.16.0, which supports the source's `MjSpec` and `mj_fullM(model, data, matrix)` interfaces. The environment checker tests those interfaces before a long run. Changes in platform, dependency version or solver behavior can affect contact simulations and timings.
