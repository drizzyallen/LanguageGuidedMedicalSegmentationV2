"""Task 0.4: run the 18 main Phase 0 trainings (3 methods x 2 datasets x 3 seeds) as a GPU queue.

One job per GPU at a time; longest jobs first. Each job's console output goes to runs/logs/<job>.log, and its
outcome to runs/queue/<job>.done or runs/queue/<job>.failed. Jobs with a .done marker are skipped, so the queue
can be restarted after an interruption. Status: runs/queue/status.log.

Start detached (survives logout):
  setsid nohup python src/run_phase0.py --gpus 7 1 2 3 > runs/queue/scheduler.out 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = (1001, 1002, 1003)
SCRIPTS = {"unet": "src/train_unet.py", "transunet": "src/train_transunet.py", "nnunet": "src/nnunet/train.py"}
# Longest first (estimates from reports/PHASE0_MODEL_SETUP.md).
ORDER = [("nnunet", "qata"), ("nnunet", "mosmed"), ("transunet", "qata"), ("unet", "qata"),
         ("transunet", "mosmed"), ("unet", "mosmed")]
QUEUE = os.path.join(ROOT, "runs", "queue")
LOGS = os.path.join(ROOT, "runs", "logs")


def status(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(os.path.join(QUEUE, "status.log"), "a") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    os.makedirs(QUEUE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    jobs = [(m, d, s) for m, d in ORDER for s in SEEDS]
    pending = [j for j in jobs if not os.path.exists(os.path.join(QUEUE, "{}_{}_seed{}.done".format(*j)))]
    status(f"scheduler start: {len(pending)} of {len(jobs)} jobs pending, GPUs {args.gpus}")

    running = {}  # gpu -> (job, process, start time)
    while pending or running:
        for gpu, (job, proc, start) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            name = "{}_{}_seed{}".format(*job)
            hours = (time.time() - start) / 3600
            marker = "done" if code == 0 else "failed"
            with open(os.path.join(QUEUE, f"{name}.{marker}"), "w") as f:
                f.write(f"exit {code}, {hours:.2f} h, GPU {gpu}\n")
            status(f"{marker.upper()} {name} on GPU {gpu} (exit {code}, {hours:.2f} h)")
            del running[gpu]
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            method, dataset, seed = job
            name = f"{method}_{dataset}_seed{seed}"
            stale = os.path.join(QUEUE, f"{name}.failed")
            if os.path.exists(stale):
                os.remove(stale)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(os.path.join(LOGS, f"{name}.log"), "w")
            proc = subprocess.Popen([sys.executable, SCRIPTS[method], "--dataset", dataset, "--seed", str(seed)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (job, proc, time.time())
            status(f"START {name} on GPU {gpu} (pid {proc.pid})")
        time.sleep(30)
    status("scheduler finished: all jobs processed")


if __name__ == "__main__":
    main()
