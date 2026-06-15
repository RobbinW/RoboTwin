import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


PROXY_UNSET = (
    "env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY "
    "-u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY"
)


def _episode_indices(directory: Path, suffix: str) -> list[int]:
    if not directory.exists():
        return []
    pattern = re.compile(rf"episode(\d+)\.{re.escape(suffix)}$")
    indices = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _contiguous_prefix(indices: list[int]) -> int:
    expected = 0
    for index in sorted(indices):
        if index == expected:
            expected += 1
        elif index > expected:
            break
    return expected


def discover_tasks(data_root: Path, output_config: str) -> list[dict]:
    tasks = []
    for task_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        traj_dir = task_dir / "demo_clean" / "_traj_data"
        traj_indices = _episode_indices(traj_dir, "pkl")
        if not traj_indices:
            continue
        output_dir = task_dir / output_config / "data"
        output_indices = _episode_indices(output_dir, "hdf5")
        target_episode_num = max(traj_indices) + 1
        done_prefix = _contiguous_prefix(output_indices)
        tasks.append(
            {
                "task": task_dir.name,
                "traj_count": len(traj_indices),
                "traj_min": min(traj_indices),
                "traj_max": max(traj_indices),
                "target_episode_num": target_episode_num,
                "output_count": len(output_indices),
                "done_prefix": done_prefix,
                "remaining": max(0, target_episode_num - done_prefix),
                "has_holes": len(traj_indices) != target_episode_num,
            }
        )
    return tasks


def assign_tasks(tasks: list[dict], gpus: list[int]) -> dict[int, list[dict]]:
    assignments = {gpu: [] for gpu in gpus}
    loads = {gpu: 0 for gpu in gpus}
    for task in sorted(tasks, key=lambda item: (-item["remaining"], item["task"])):
        gpu = min(gpus, key=lambda candidate: loads[candidate])
        assignments[gpu].append(task)
        loads[gpu] += task["remaining"]
    return assignments


def write_worker_script(
    *,
    repo_root: Path,
    run_dir: Path,
    gpu: int,
    tasks: list[dict],
    python_bin: str,
    config: str,
) -> Path:
    script_path = run_dir / f"worker_gpu{gpu}.sh"
    log_dir = run_dir / f"gpu{gpu}_logs"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"echo '[worker gpu{gpu}] start '$(date)",
    ]
    for task in tasks:
        task_name = task["task"]
        episode_num = int(task["target_episode_num"])
        task_log = log_dir / f"{task_name}.log"
        lines.extend(
            [
                f"echo '[worker gpu{gpu}] task={task_name} episode_num={episode_num} start '$(date)",
                (
                    f"CUDA_VISIBLE_DEVICES={gpu} {PROXY_UNSET} {shlex.quote(python_bin)} "
                    f"script/collect_data.py {shlex.quote(task_name)} {shlex.quote(config)} "
                    f"--episode_num {episode_num} > {shlex.quote(str(task_log))} 2>&1"
                ),
                "status=$?",
                f"echo '[worker gpu{gpu}] task={task_name} exit='$status' end '$(date)",
                f"echo -e '{task_name}\\t'$status'\\t'$(date +%s) >> {shlex.quote(str(run_dir / f'gpu{gpu}_status.tsv'))}",
                "if [ $status -ne 0 ]; then",
                f"  echo '[worker gpu{gpu}] task={task_name} FAILED, see {task_log}'",
                "fi",
            ]
        )
    lines.append(f"echo '[worker gpu{gpu}] done '$(date)")
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return script_path


def launch(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    data_root = repo_root / "data"
    run_dir = Path(args.run_dir or (repo_root / "logs" / f"pointworld_batch_{time.strftime('%Y%m%d_%H%M%S')}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(data_root, args.config)
    pending = [task for task in tasks if task["remaining"] > 0]
    gpus = [int(item) for item in args.gpus.split(",") if item]
    assignments = assign_tasks(pending, gpus)

    plan = {
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "config": args.config,
        "python_bin": args.python_bin,
        "gpus": gpus,
        "tasks_total": len(tasks),
        "episodes_total": sum(task["target_episode_num"] for task in tasks),
        "pending_tasks": len(pending),
        "pending_episodes": sum(task["remaining"] for task in pending),
        "assignments": {str(gpu): assignments[gpu] for gpu in gpus},
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(json.dumps({k: plan[k] for k in ("tasks_total", "episodes_total", "pending_tasks", "pending_episodes")}, indent=2))
    print(f"run_dir={run_dir}")
    for gpu in gpus:
        worker_script = write_worker_script(
            repo_root=repo_root,
            run_dir=run_dir,
            gpu=gpu,
            tasks=assignments[gpu],
            python_bin=args.python_bin,
            config=args.config,
        )
        session_name = f"{args.session_prefix}_gpu{gpu}"
        subprocess.run(["tmux", "kill-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, str(worker_script)], check=True)
        print(f"launched {session_name}: tasks={len(assignments[gpu])}, episodes={sum(t['remaining'] for t in assignments[gpu])}")


def status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    plan = json.loads((run_dir / "plan.json").read_text())
    for gpu in plan["gpus"]:
        status_path = run_dir / f"gpu{gpu}_status.tsv"
        done = []
        failed = []
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    done.append(parts)
                    if parts[1] != "0":
                        failed.append(parts)
        assigned = plan["assignments"][str(gpu)]
        print(f"gpu{gpu}: done_tasks={len(done)}/{len(assigned)}, failed={len(failed)}")
        if failed:
            for item in failed:
                print("  FAILED", "\t".join(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--repo_root", default="/data/dex/RoboTwin")
    launch_parser.add_argument("--config", default="pointworld_behavior_compact_head")
    launch_parser.add_argument("--gpus", default="2,3,4,5")
    launch_parser.add_argument("--python_bin", default="/data/dex/conda-envs/RoboTwin/bin/python")
    launch_parser.add_argument("--session_prefix", default="robotwin_pw_batch")
    launch_parser.add_argument("--run_dir", default=None)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "launch":
        launch(parsed)
    elif parsed.command == "status":
        status(parsed)
