"""Check dependencies, compiled dynamics and one closed-loop integration step."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true", help="skip model, dynamics and controller checks")
    parser.add_argument("--render", action="store_true", help="also render one small frame to test the graphics backend")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="allow checks without the optional video encoder")
    args = parser.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    sys.path.insert(0, str(ROOT))
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Rendering backend: {os.environ['MUJOCO_GL']}")
    failed = False
    for package, module in (("numpy", "numpy"), ("scipy", "scipy"), ("mujoco", "mujoco"),
                            ("daqp", "daqp"), ("matplotlib", "matplotlib"), ("Pillow", "PIL")):
        try:
            importlib.import_module(module)
            print(f"OK {package}: {metadata.version(package)}")
        except (ImportError, OSError, metadata.PackageNotFoundError) as error:
            failed = True
            print(f"FAILED {package}: {error}")
    executable = shutil.which("ffmpeg")
    if executable:
        try:
            completed = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True,
                                       text=True, check=True, timeout=20)
            if "libx264" not in completed.stdout:
                raise RuntimeError("FFmpeg has no libx264 encoder.")
            print(f"OK FFmpeg/libx264: {executable}")
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            failed = True
            print(f"FAILED FFmpeg: {error}")
    elif not args.skip_ffmpeg:
        failed = True
        print("FAILED FFmpeg: executable not found on PATH.")
    else:
        print("SKIPPED FFmpeg")
    asset = ROOT / "assets" / "apollo" / "apptronik_apollo.xml"
    if asset.is_file():
        print(f"OK Apollo model: {asset}")
    else:
        failed = True
        print(f"FAILED Apollo model: missing {asset}")
    if not failed and not args.imports_only:
        try:
            import mujoco
            import numpy as np
            from teamimp import mission, textures
            with tempfile.TemporaryDirectory(prefix="team-impedance-check-") as temporary:
                previous_textures = textures.TEX_DIR
                textures.TEX_DIR = temporary
                try:
                    test_mission = mission.Mission(mission.Cfg(
                        n=2, profile="AL 280", background=False, verbose=False, kz=0.0))
                    model, data = test_mission.m, test_mission.d
                    test_mission.tm.update(data.qpos.copy(), data.qvel.copy(), dyn=True)
                    print("OK compiled dynamics interface: TeamModel.update(dyn=True)")
                    test_mission.R.step(1)
                    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                            and np.isfinite(data.ctrl).all()):
                        raise RuntimeError("The closed-loop step produced a nonfinite state or control.")
                    print(f"OK two-robot controller and integration: nq={model.nq}, nv={model.nv}; "
                          f"simulated {test_mission.R.t:.3f} s")
                    if args.render:
                        with mujoco.Renderer(model, height=120, width=160) as renderer:
                            renderer.update_scene(data)
                            frame = renderer.render()
                        if frame.shape != (120, 160, 3):
                            raise RuntimeError(f"Unexpected image dimensions: {frame.shape}")
                        print("OK off-screen rendering: 160 x 120")
                finally:
                    textures.TEX_DIR = previous_textures
        except Exception as error:
            failed = True
            print(f"FAILED model/graphics check: {type(error).__name__}: {error}")
    if failed:
        print("Environment check failed. Resolve the failures above before running the study.")
        return 1
    print("Environment check passed. Full missions and paper conclusions were not tested by this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
