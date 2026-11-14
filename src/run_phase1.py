"""Task 1.6: run the 18 main Phase 1 experiments (3 methods x 2 datasets x 3 seeds) as a GPU queue.

Each job is src/phase1/run_job.sh: training -> test predictions of the best checkpoint -> common evaluation.
One job per GPU at a time; longest jobs first. Console output: runs/logs/<job>.log; outcome:
runs/queue_phase1/<job>.done or .failed. Jobs with a .done marker are skipped, so the queue can be restarted.
A job that is started again (after a failure or an interruption) starts from scratch: its old run directory is
moved to runs/interrupted/<job>_<timestamp>/ first. Status: runs/queue_phase1/status.log.

Start detached (survives logout):
  setsid nohup python3 src/run_phase1.py --gpus 0 1 2 3 > runs/queue_phase1/scheduler.out 2>&1 &
"""
import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = (1001, 1002, 1003)
# Longest first (estimates from reports/PHASE1_MODEL_SETUP.md, compute estimate).
ORDER = [("lvit", "qata"), ("reclmis", "qata"), ("lvit", "mosmed"), ("ddcmd", "qata"), ("reclmis", "mosmed"),
         ("ddcmd", "mosmed")]
QUEUE = os.path.join(ROOT, "runs", "queue_phase1")
LOGS = os.path.join(ROOT, "runs", "logs")


def status(msg):
    line = "{:%Y-%m-%d %H:%M:%S} {}".format(datetime.now(), msg)
    print(line, flush=True)
    with open(os.path.join(QUEUE, "status.log"), "a") as f:
        f.write(line + "\n")


def job_name(job):
    return "{}_{}_seed{}".format(*job)


def clear_old_run(job):
    run = os.path.join(ROOT, "runs", job[0], job[1], "seed_{}".format(job[2]))
    if os.path.exists(run):
        dest = os.path.join(ROOT, "runs", "interrupted", "{}_{:%Y%m%d_%H%M%S}".format(job_name(job), datetime.now()))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(run, dest)
        status("moved old run directory of {} to {}".format(job_name(job), os.path.relpath(dest, ROOT)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these job names (default: all 18)")
    args = parser.parse_args()
    os.makedirs(QUEUE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    jobs = [(m, d, s) for m, d in ORDER for s in SEEDS]
    if args.only is not None:
        jobs = [j for j in jobs if job_name(j) in args.only]
    pending = [j for j in jobs if not os.path.exists(os.path.join(QUEUE, job_name(j) + ".done"))]
    status("scheduler start: {} of {} jobs pending, GPUs {}".format(len(pending), len(jobs), args.gpus))

    running = {}  # gpu -> (job, process, start time)
    while pending or running:
        for gpu, (job, proc, start) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            hours = (time.time() - start) / 3600
            marker = "done" if code == 0 else "failed"
            with open(os.path.join(QUEUE, "{}.{}".format(job_name(job), marker)), "w") as f:
                f.write("exit {}, {:.2f} h, GPU {}\n".format(code, hours, gpu))
            status("{} {} on GPU {} (exit {}, {:.2f} h)".format(marker.upper(), job_name(job), gpu, code, hours))
            del running[gpu]
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            name = job_name(job)
            stale = os.path.join(QUEUE, name + ".failed")
            if os.path.exists(stale):
                os.remove(stale)
            clear_old_run(job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(os.path.join(LOGS, name + ".log"), "w")
            proc = subprocess.Popen(["bash", "src/phase1/run_job.sh", job[0], job[1], str(job[2])], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (job, proc, time.time())
            status("START {} on GPU {} (pid {})".format(name, gpu, proc.pid))
        time.sleep(30)
    status("scheduler finished: all jobs processed")


if __name__ == "__main__":
    main()
