"""Run the aluminium-girder study in an isolated, resumable working directory."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = "results/mission/state_lift_al_n4_kz0.pkl"


@dataclass(frozen=True)
class Job:
    name: str
    args: tuple[str, ...]
    outputs: tuple[str, ...] = ()
    reuse_existing: bool = False


def experiment(name: str, script: str, *args: str, outputs=(), reuse=False) -> Job:
    return Job(name, (f"experiments/{script}.py", *map(str, args)), tuple(outputs), reuse)


def mission(tag: str, n: int = 4, *args: str) -> Job:
    return experiment(
        tag, "mission_run", "--n", str(n), "--tag", tag,
        *args, outputs=(f"results/mission/{tag}.npz", f"results/mission/{tag}_logs.pkl"),
        reuse=True,
    )


def main_completion() -> Job:
    # The same completed-main-run gate used by experiments/run_aluminium.sh.
    return Job("check_main_completed", ("-c", "import numpy as np, sys; "
               "sys.exit(int(np.load('results/mission/al_n4_kz0.npz')['failed'][0] >= 0))"))


def video_missions() -> list[Job]:
    # These are the three tags read by the current video renderer.
    return [mission("al_n4_kz0", 4, "--kz", "0"), main_completion(),
            mission("al_n3_kz0", 3, "--kz", "0"),
            mission("al_n4_noshare_kz0", 4, "--no-share-internal", "--kz", "0")]


def exact_jobs() -> list[Job]:
    jobs = [experiment(f"exact_{name}", "exact_validation", name)
            for name in ("identities", "rates", "antagonism", "refdisp", "decoupling", "corollary")]
    return jobs + [experiment(f"exact_zoh_{period}", "exact_validation", "zoh", period)
                   for period in ("4", "2", "1", "0.5")]


def video_job() -> Job:
    return Job("render_video", ("video/make_video.py", "all"),
               ("video/team_impedance_mission.mp4",))


def aluminium_jobs() -> list[Job]:
    # Command order and arguments follow experiments/run_aluminium.sh.
    jobs = [mission("al_n4_kz0", 4, "--kz", "0"), main_completion(),
            experiment("save_lift_state", "save_state", "--n", "4", "--kz", "0",
                       outputs=(STATE,), reuse=True),
            experiment("push", "mujoco_push", STATE, "100"),
            experiment("push_nofriction", "mujoco_push", STATE, "100", "nofriction"),
            experiment("bias_sweep", "mujoco_bias_sweep", STATE, "63.07,58,53,48,43,38"),
            experiment("timing", "timing", STATE, "3"),
            experiment("capacity_main", "capacity", "63.07", "4,3,2"),
            mission("al_n4", 4, "--kz", "1"),
            mission("al_n4_ff", 4, "--internal", "feedforward"),
            mission("al_n4_noshare_kz0", 4, "--no-share-internal", "--kz", "0"),
            mission("al_n3_kz0", 3, "--kz", "0"),
            mission("al_n2_kz0", 2, "--kz", "0")]
    for noise, seed in ((5, 1), (15, 2)):
        jobs.append(mission(f"al_n4_noise{noise}", 4, "--kz", "0", "--noise-f", str(noise),
                            "--noise-m", str(noise / 10), "--noise-seed", str(seed)))
    for tag, flags in (("al_n4_kin_kz0", ("--kz", "0")),
                       ("al_n4_kin", ("--kz", "1")),
                       ("al_n4_kin_ff", ("--internal", "feedforward"))):
        jobs.append(mission(tag, 4, "--kin-err", "0.001", "--kin-err-seed", "3", *flags))
    for mass in (70, 66, 60, 55, 50, 40):
        jobs.append(mission(f"al_n4_m{mass}", 4, "--kz", "1", "--ctrl-mass", str(mass)))
    for mass in (70, 66, 60, 55, 50, 40, 30):
        jobs.append(mission(f"al_n4_m{mass}_kz0", 4, "--kz", "0", "--ctrl-mass", str(mass)))
    jobs.append(mission("al_n4_m70_ff", 4, "--internal", "feedforward", "--ctrl-mass", "70"))
    return jobs


def contact_jobs() -> list[Job]:
    # Command order and amplitudes follow experiments/run_contact_tests.sh.
    jobs = [experiment(f"refdisp_{a}", "mujoco_refdisp", STATE, str(a)) for a in (15, 25, 40)]
    jobs.append(experiment("refdisp_25_nofriction", "mujoco_refdisp", STATE, "25", "nofriction"))
    jobs += [experiment(f"decoupling_{a}", "mujoco_decoupling", STATE, str(a)) for a in (15, 30, 40, 60)]
    jobs += [experiment(f"decoupling_{a}_nofriction", "mujoco_decoupling", STATE, str(a), "nofriction")
             for a in (15, 30)]
    return jobs


def sweep_jobs() -> list[Job]:
    # The 183.36 kg steel runs are intentional capacity-sweep comparisons in this paper.
    jobs = [mission(f"n{n}_kz0", n, "--mass", "183.36", "--kz", "0") for n in (4, 3, 2)]
    for n, mass in ((4, 240), (4, 260), (4, 280), (3, 120), (3, 140), (3, 160),
                    (2, 100), (2, 120), (2, 140)):
        jobs.append(mission(f"sweep_n{n}_{mass}", n, "--mass", str(mass), "--kz", "0"))
    return jobs


def full_jobs() -> list[Job]:
    # run_all.sh: aluminium, contacts, sweep, exact checks, metrics, both bias analyses, trajectory.
    jobs = aluminium_jobs() + contact_jobs() + sweep_jobs() + exact_jobs()
    jobs += [experiment("mission_metrics", "mission_metrics"),
             experiment("model_error_offset", "model_error_offset"),
             experiment("model_error_offset_kz0", "model_error_offset", "kz0"),
             experiment("extract_trajectory", "extract_traj", "al_n4_kz0")]
    # Original README's subsequent figure/video commands.
    jobs += [experiment("render_teaser", "render_teaser"),
             experiment("render_stills", "render_stills"),
             experiment("all_figures", "figures", "all"), video_job()]
    return jobs


def make_plan(task: str) -> list[Job]:
    if task == "main":
        return video_missions()[:2]
    if task == "exact":
        return exact_jobs()
    if task == "figures":
        return [experiment("figure_exact", "figures", "exact"),
                experiment("figure_contact", "figures", "contact")]
    if task == "capacity":
        return [experiment("capacity_grid", "capacity",
                           "60,63.07,80,100,120,140,160,180,183,183.36,200,220,240,260,280,300", "4,3,2")]
    if task == "video":
        return video_missions() + [video_job()]
    return full_jobs()


def prepare_work_dir(work: Path) -> None:
    if work == ROOT or ROOT in work.parents:
        raise ValueError("The working directory must be outside the repository.")
    marker = work / ".reproduction-workspace.json"
    if work.exists() and any(work.iterdir()) and not marker.exists():
        raise ValueError(f"Refusing to use an existing nonempty directory without a workspace marker: {work}")
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source") != str(ROOT):
            raise ValueError("This working directory belongs to a different source repository.")
    first_copy = not marker.exists()
    work.mkdir(parents=True, exist_ok=True)
    ignored = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "frames", "*.mp4", "*.pkl")
    for name in ("teamimp", "assets", "experiments", "video", "paper"):
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, work / name, dirs_exist_ok=True, ignore=ignored)
    if first_copy:
        shutil.copytree(ROOT / "results", work / "results", dirs_exist_ok=True, ignore=ignored)
    (work / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    (work / ".completed").mkdir(exist_ok=True)
    marker.write_text(json.dumps({"source": str(ROOT)}, indent=2) + "\n", encoding="utf-8")


def command_text(args: tuple[str, ...]) -> str:
    command = [sys.executable, *args]
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Tasks: main = one full mission; exact = ten compiled-model checks;\n"
                                     "figures = exact/contact plots from supplied data; video = three missions\n"
                                     "and the supplementary video; full = all 37 supplied mission variants,\n"
                                     "compiled-model checks, contact tests, plots and video.\n"
                                     "Simulations run sequentially. Full reproduction can take many hours\n"
                                     "and writes several GB of logs. Published result files are preserved.")
    parser.add_argument("task", choices=("main", "exact", "figures", "capacity", "video", "full"))
    parser.add_argument("--work-dir", type=Path, default=ROOT.parent / ".reproduction_work",
                        help="isolated working copy, outside this repository (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print every command without writing or running anything")
    parser.add_argument("--force", action="store_true", help="rerun completed jobs in the working copy")
    args = parser.parse_args()
    work = args.work_dir.expanduser().resolve()
    jobs = make_plan(args.task)
    print(f"Working copy: {work}", flush=True)
    print(f"Plan: {len(jobs)} sequential jobs", flush=True)
    if args.dry_run:
        for number, job in enumerate(jobs, 1):
            print(f"{number:02d}. {command_text(job.args)}")
        return 0
    prepare_work_dir(work)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    env.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    for number, job in enumerate(jobs, 1):
        marker = work / ".completed" / f"{job.name}.json"
        outputs_exist = bool(job.outputs) and all((work / name).is_file() for name in job.outputs)
        if not args.force and ((job.reuse_existing and outputs_exist) or
                               (marker.exists() and (not job.outputs or outputs_exist))):
            print(f"[{number}/{len(jobs)}] Reuse {job.name}", flush=True)
            continue
        log = work / "logs" / f"{job.name}.log"
        print(f"[{number}/{len(jobs)}] {command_text(job.args)}", flush=True)
        print(f"  Log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, *job.args], cwd=work, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps({"command": list(job.args)}, indent=2) + "\n", encoding="utf-8")
    if args.task in ("video", "full"):
        source = work / "video" / "team_impedance_mission.mp4"
        destination = ROOT / "video" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Video copied to: {destination}", flush=True)
    print(f"Finished. Generated results and figures: {work}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Reproduction stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
