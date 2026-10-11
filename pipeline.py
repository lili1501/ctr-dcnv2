#!/usr/bin/env python
"""One command for the whole project: data -> Spark -> LightGBM -> DCN-V2 (+ablation) -> export -> report -> resume numbers.

    python pipeline.py                                  # real Criteo at data/raw/train.txt, every 4th row (~11.5M rows)
    python pipeline.py --subsample-every 40             # quick check (~1.1M rows) before the long run
    python pipeline.py --subsample-every 1 --seeds "0 1 2"   # all rows (needs far more memory)

It wraps scripts/run_pipeline.sh (the single source of truth for the stages), then writes
  reports/results.md, results.json, calibration.png   the comparison
  reports/run_info.json                                what was run and how long it took
  reports/resume_numbers.md                            measured facts + a draft bullet, from the results only
Any option left out falls back to the same environment variable run_pipeline.sh reads (SUBSAMPLE_EVERY, SEEDS, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _cgroup_limit_gb():
    """Memory limit of this job/session (cgroup v2, then v1), in GiB, or None if unlimited/unknown."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < (1 << 50):
            return int(v) / 2 ** 30
    return None


def available_memory_gb(env=None, meminfo="/proc/meminfo"):
    """The memory this process may really use: the smallest of the SLURM allocation, the cgroup limit and MemAvailable.
    (/proc/meminfo alone shows the whole node, which is far more than a scheduler-limited session may use.)"""
    env = os.environ if env is None else env
    cands = []
    if env.get("SLURM_MEM_PER_NODE", "").isdigit():
        cands.append(int(env["SLURM_MEM_PER_NODE"]) / 1024)
    c = _cgroup_limit_gb()
    if c:
        cands.append(c)
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    cands.append(int(line.split()[1]) / 2 ** 20)
    except OSError:
        pass
    return min(cands) if cands else None


def cuda_env(env, root: str, site_dirs=None) -> dict:
    """Settings that let pip-installed TensorFlow (tensorflow[and-cuda]) find its CUDA libraries: the nvidia/*/lib
    directories of the active environment go first on LD_LIBRARY_PATH (on HiPerGator TensorFlow otherwise reports no GPU),
    and the CUDA kernel cache stays inside the project folder instead of ~/.nv. Returns {} when no such libraries exist."""
    import glob
    import site
    dirs = []
    for sp in (site_dirs if site_dirs is not None else site.getsitepackages()):
        dirs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    if not dirs:
        return {}
    out = {"LD_LIBRARY_PATH": ":".join(dirs + [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p not in dirs])}
    if "CUDA_CACHE_PATH" not in env:
        out["CUDA_CACHE_PATH"] = os.path.join(root, "data", "cuda_cache")
    return out


def spark_defaults(env, mem_gb, cpus: int, root: str) -> dict:
    """Spark settings to add when the user has not set them: ~60% of available memory for the driver (the later
    TensorFlow/LightGBM stages run after Spark exits), all allocated cores, shuffle files inside the project folder."""
    out = {}
    if "SPARK_DRIVER_MEMORY" not in env and mem_gb:
        out["SPARK_DRIVER_MEMORY"] = f"{max(4, int(mem_gb * 0.6))}g"
    if "SPARK_CORES" not in env:
        out["SPARK_CORES"] = str(max(1, cpus))
    if "SPARK_LOCAL_DIRS" not in env and "SPARK_LOCAL_DIR" not in env:
        out["SPARK_LOCAL_DIRS"] = os.path.join(root, "data", "spark_tmp")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["kaggle", "attribution"], default="kaggle",
                    help="kaggle: Criteo Display Advertising train.txt; attribution: Criteo Attribution Modeling "
                         "(criteo_attribution_dataset.tsv.gz from Hugging Face)")
    ap.add_argument("--max-rows", type=int, help="attribution only: use just the first N rows (the earliest in time); "
                    "a partial download is fine")
    ap.add_argument("--raw", help="input file (default data/raw/train.txt, or data/raw/criteo_attribution_dataset.tsv.gz)")
    ap.add_argument("--subsample-every", type=int, help="keep every Nth row (default 4; 1 = all rows)")
    ap.add_argument("--seeds", help='DCN-V2 seeds, e.g. "0 1 2" (default "0 1 2")')
    ap.add_argument("--num-buckets", type=int, help="hash buckets per categorical feature (default 200000)")
    ap.add_argument("--processed"); ap.add_argument("--artifacts"); ap.add_argument("--reports")
    ap.add_argument("--stages", help="subset of: gen prep lgbm dcn export compare")
    ap.add_argument("--synthetic-rows", type=int, help="TESTING ONLY: generate this many synthetic rows instead of using Criteo")
    a = ap.parse_args(argv)

    env = os.environ.copy()
    attribution = a.dataset == "attribution"
    env["CTR_SCHEMA"] = "criteo_attribution" if attribution else "criteo_kaggle"
    defaults = {"RAW": "data/raw/criteo_attribution_dataset.tsv.gz" if attribution else "data/raw/train.txt",
                "SUBSAMPLE_EVERY": "1" if attribution else "4", "SEEDS": "0 1 2", "NUM_BUCKETS": "200000",
                "PROCESSED": "data/processed", "REPORTS": "reports"}
    opts = {"RAW": a.raw, "SUBSAMPLE_EVERY": a.subsample_every, "SEEDS": a.seeds, "NUM_BUCKETS": a.num_buckets,
            "PROCESSED": a.processed, "ART": a.artifacts, "REPORTS": a.reports, "STAGES": a.stages,
            "SYNTHETIC_ROWS": a.synthetic_rows}
    for k, v in opts.items():
        if v is not None:
            env[k] = str(v)
    if a.synthetic_rows:                       # synthetic runs use the whole file, no subsampling, unless asked
        env.setdefault("SUBSAMPLE_EVERY", "")
        if a.subsample_every is None:
            env["SUBSAMPLE_EVERY"] = ""
    else:
        env.setdefault("SUBSAMPLE_EVERY", defaults["SUBSAMPLE_EVERY"])
        env.setdefault("NUM_BUCKETS", defaults["NUM_BUCKETS"])
    if a.subsample_every == 1:
        env["SUBSAMPLE_EVERY"] = ""

    if attribution and a.synthetic_rows:
        sys.stderr.write("ERROR: --synthetic-rows only exists for the kaggle schema\n")
        return 2
    if attribution and a.subsample_every is None and "SUBSAMPLE_EVERY" not in os.environ:
        env["SUBSAMPLE_EVERY"] = ""            # the attribution set is small enough to use whole (16.5M rows)
    raw = env.get("RAW", defaults["RAW"])
    stages = env.get("STAGES", "gen prep lgbm dcn export compare").split()
    if not a.synthetic_rows and "prep" in stages and not attribution and not os.path.exists(os.path.join(ROOT, raw) if not os.path.isabs(raw) else raw):
        sys.stderr.write(f"ERROR: {raw} not found. Download Criteo first (see README, 'Running on real Criteo'), "
                         f"or pass --raw /path/to/train.txt\n")
        return 2

    reports = env.get("REPORTS", defaults["REPORTS"])
    processed = env.get("PROCESSED", defaults["PROCESSED"])
    artifacts = env.get("ART", "artifacts")

    sys.path.insert(0, ROOT)
    audit = None
    if attribution and "prep" in stages:
        # the dataset ships as a compressed TSV with a header and many extra columns; convert once to the pipeline's raw format
        converted = os.path.join(ROOT, "data", "raw", "criteo_attribution.txt")
        src = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
        if not os.path.exists(src):
            sys.stderr.write(f"ERROR: {raw} not found. Download criteo_attribution_dataset.tsv.gz from "
                             f"https://huggingface.co/datasets/criteo/criteo-attribution-dataset (Files and versions), "
                             f"or pass --raw /path/to/it\n")
            return 2
        if os.path.abspath(src) != os.path.abspath(converted):
            from ctr.prepare_attribution import convert
            print(f"Converting {raw} -> data/raw/criteo_attribution.txt (one pass over the file) ...", flush=True)
            audit = convert(src, converted, max_rows=a.max_rows)
            with open(converted + ".audit.json", "w") as f:
                json.dump(audit, f, indent=2)
            print(f"  {audit['rows']:,} rows, click rate {audit['click_rate']:.2%}, sorted by timestamp: {audit['sorted_by_timestamp']}")
            if not audit["sorted_by_timestamp"]:
                sys.stderr.write("ERROR: the file is not sorted by timestamp, so a positional split would not be a time split.\n")
                return 2
        env["RAW"] = raw = converted
    elif attribution and os.path.exists(os.path.join(ROOT, "data", "raw", "criteo_attribution.txt.audit.json")):
        audit = json.load(open(os.path.join(ROOT, "data", "raw", "criteo_attribution.txt.audit.json")))

    sys.path.insert(0, ROOT)
    from ctr.resources import available_cpus
    extra = spark_defaults(env, available_memory_gb(env), available_cpus(), ROOT)
    for k, v in extra.items():
        env[k] = v
    if "SPARK_LOCAL_DIRS" in extra:
        os.makedirs(extra["SPARK_LOCAL_DIRS"], exist_ok=True)
    cuda = cuda_env(env, ROOT)
    for k, v in cuda.items():
        env[k] = v
    if "CUDA_CACHE_PATH" in cuda:
        os.makedirs(cuda["CUDA_CACHE_PATH"], exist_ok=True)
    print("CUDA libraries from the Python environment: " + ("added to LD_LIBRARY_PATH" if cuda else "none found (CPU run)"), flush=True)
    print("Spark settings: " + ", ".join(f"{k}={env.get(k)}" for k in ("SPARK_DRIVER_MEMORY", "SPARK_CORES", "SPARK_LOCAL_DIRS")), flush=True)

    t0 = time.time()
    rc = subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_pipeline.sh")], env=env, cwd=ROOT).returncode
    if rc != 0:
        sys.stderr.write(f"ERROR: run_pipeline.sh failed (exit {rc}); no results written\n")
        return rc

    if "compare" in stages:
        os.makedirs(os.path.join(ROOT, reports), exist_ok=True)
        info = {"subsample_every": int(env["SUBSAMPLE_EVERY"]) if env.get("SUBSAMPLE_EVERY") else 1,
                "seeds": env.get("SEEDS", defaults["SEEDS"]), "num_buckets": env.get("NUM_BUCKETS") or None,
                "synthetic_rows": a.synthetic_rows, "raw": raw, "elapsed_seconds": round(time.time() - t0, 1),
                "stages": stages,
                "dataset": a.dataset, "max_rows": a.max_rows, "attribution_audit": audit}
        with open(os.path.join(ROOT, reports, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        sys.path.insert(0, ROOT)
        from ctr.resume_summary import write_summary
        path = write_summary(os.path.join(ROOT, reports), os.path.join(ROOT, processed), os.path.join(ROOT, artifacts))
        print(f"\nResults written to {reports}/ . Resume numbers: {os.path.relpath(path, ROOT)}\n")
        print(open(path).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
