#!/usr/bin/env python3
"""
Overnight auto-search script based on the v55 main family.

Plan
----
3 workers × ~10 hours × ~4-6 submissions/hour ≈ 120-180 total runs.
This script intentionally searches a bit wider than the first v55 script,
but still stays inside the validated family:
- no read-path changes
- no new kernel family
- no 16x128 branch resurrection
- no unroll / direct-gather experiments

Search priorities
-----------------
P1: M256_K1536_N3072  (main bottleneck, widest search)
P2: M64_K2048_N7168   (secondary bottleneck)
P3: M16_K7168_N2112   (small, careful neighborhood)
P4: K=512 shapes      (light opportunistic tuning)

Usage
-----
Dry run:
  python3 auto_search1_v55_3workers.py --worker 1 --dry-run

Overnight:
  HOME=/mnt/d/code/match/amd/gpu_kernel/auto/worker_1 python3 auto_search1_v55_3workers.py --worker 1
  HOME=/mnt/d/code/match/amd/gpu_kernel/auto/worker_2 python3 auto_search1_v55_3workers.py --worker 2
  HOME=/mnt/d/code/match/amd/gpu_kernel/auto/worker_3 python3 auto_search1_v55_3workers.py --worker 3

Analyze:
  python3 auto_search1_v55_3workers.py --analyze
"""

import argparse
import ast
import copy
import json
import math
import os
import re
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BASELINES = {
    "M4_K512_N2880": 6.17,
    "M16_K7168_N2112": 10.10,
    "M32_K512_N4096": 6.93,
    "M32_K512_N2880": 6.85,
    "M64_K2048_N7168": 12.70,
    "M256_K1536_N3072": 12.30,
}

SHAPES = {
    0: {
        "label": "M4_K512_N2880",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json",
        "tier": "M_LEQ_8",
        "priority": 4,
    },
    1: {
        "label": "M16_K7168_N2112",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json",
        "tier": "M_LEQ_32",
        "priority": 3,
    },
    2: {
        "label": "M32_K512_N4096",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=4096-K=512.json",
        "tier": "M_LEQ_32",
        "priority": 4,
    },
    3: {
        "label": "M32_K512_N2880",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json",
        "tier": "M_LEQ_32",
        "priority": 4,
    },
    4: {
        "label": "M64_K2048_N7168",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=7168-K=2048.json",
        "tier": "M_LEQ_64",
        "priority": 2,
    },
    5: {
        "label": "M256_K1536_N3072",
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=3072-K=1536.json",
        "tier": "M_LEQ_256",
        "priority": 1,
    },
}

TARGET_GUARD = {
    "M256_K1536_N3072": 0.40,
    "M64_K2048_N7168": 0.25,
    "M16_K7168_N2112": 0.30,
    "M4_K512_N2880": 0.15,
    "M32_K512_N4096": 0.20,
    "M32_K512_N2880": 0.20,
}
NON_TARGET_SOFT_REGRESSION = 0.20
NON_TARGET_HARD_REGRESSION = 0.35


def baseline_geo_mean():
    vals = list(BASELINES.values())
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_shape_configs(base_path: Path):
    text = base_path.read_text()
    m = re.search(r"SHAPE_CONFIGS\s*=\s*(\{[\s\S]*?\})\n\n# ===", text)
    if not m:
        raise RuntimeError("Could not locate SHAPE_CONFIGS block")
    return ast.literal_eval(m.group(1)), text


def make_submission(base_code: str, configs: dict) -> str:
    start = base_code.index("SHAPE_CONFIGS = {")
    end = base_code.index("\n\n# ===", start)
    cfg_str = json.dumps(configs, indent=4)
    cfg_str = cfg_str.replace(": null", ": None").replace(": true", ": True").replace(": false", ": False")
    return base_code[:start] + "SHAPE_CONFIGS = " + cfg_str + base_code[end:]


TRANSIENT_ERROR_PATTERNS = [
    r"Rate limit exceeded",
    r"dns error",
    r"failed to lookup address information",
    r"error trying to connect",
    r"Service Unavailable",
    r"status 5\d\d",
    r"502 Bad Gateway",
    r"503 Service Unavailable",
    r"504 Gateway Timeout",
    r"connection reset",
    r"connection refused",
    r"timed out",
    r"timeout",
    r"temporary failure",
    r"network is unreachable",
    r"TLS",
    r"transport error",
    r"error sending request",
    r"connection aborted",
    r"EOF while parsing a value",
    r"unexpected end of file",
]

FATAL_ERROR_PATTERNS = [
    r"ImportError",
    r"SyntaxError",
    r"Traceback",
    r"another stream",
    r"Server processing error",
    r"invalid submission",
    r"custom_kernel",
]


def classify_submit_output(output: str) -> str:
    m = re.search(r"Rate limit exceeded.*?Try again in (\d+)s", output)
    if m:
        return "rate_limit"
    lower = output.lower()
    for pat in FATAL_ERROR_PATTERNS:
        if re.search(pat, output, re.I):
            return "fatal"
    for pat in TRANSIENT_ERROR_PATTERNS:
        if re.search(pat, output, re.I):
            return "transient"
    if parse_results(output):
        return "ok"
    if "Application error" in output or "Waiting for results" in output:
        return "transient"
    return "fatal"


def submit(submission_path: Path, worker_home: Path, max_attempts: int = 6):
    env = os.environ.copy()
    env["HOME"] = str(worker_home.resolve())
    cmd = [
        "popcorn-cli", "submit",
        "--mode", "benchmark",
        "--gpu", "MI355X",
        "--leaderboard", "amd-mxfp4-mm",
        str(submission_path.resolve()),
        "--no-tui",
    ]

    last_output = ""
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=780, env=env)
            output = (r.stdout or "") + "\n" + (r.stderr or "")
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: subprocess.run timed out"
        except Exception as e:
            output = f"ERROR: {e}"

        status = classify_submit_output(output)
        last_output = output

        if status == "ok":
            return {"status": "ok", "attempts": attempts, "raw": output}

        if status == "rate_limit":
            m = re.search(r"Rate limit exceeded.*?Try again in (\d+)s", output)
            wait_s = int(m.group(1)) + 10 if m else 70
            print(f"  Rate limited on attempt {attempts}, sleep {wait_s}s")
            time.sleep(wait_s)
            continue

        if status == "transient":
            wait_s = min(180, 15 * attempts + random.randint(3, 12))
            print(f"  Transient submit error on attempt {attempts}/{max_attempts}, retry in {wait_s}s")
            time.sleep(wait_s)
            continue

        return {"status": "fatal", "attempts": attempts, "raw": output}

    return {"status": "defer", "attempts": attempts, "raw": last_output}


def parse_results(output: str):
    results = {}
    for m in re.finditer(r"k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+).*?\n[^⏱]*⏱\s*([\d.]+)", output, re.S):
        k, mv, n, t = m.groups()
        results[f"M{mv}_K{k}_N{n}"] = float(t)
    return results


def geo_mean(results: dict) -> float:
    if not results:
        return 999.0
    vals = list(results.values())
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def weighted_score(results: dict, target: str) -> float:
    """Lower is better."""
    if not results or target not in results:
        return 1e9

    pri_w = {
        "M256_K1536_N3072": 1.00,
        "M64_K2048_N7168": 0.95,
        "M16_K7168_N2112": 0.85,
        "M4_K512_N2880": 0.70,
        "M32_K512_N4096": 0.70,
        "M32_K512_N2880": 0.70,
    }[target]

    score = pri_w * results[target] + 0.40 * geo_mean(results)
    target_reg = results[target] - BASELINES[target]
    if target_reg > TARGET_GUARD[target]:
        score += 10.0 + 12.0 * target_reg

    for shape, base in BASELINES.items():
        if shape == target or shape not in results:
            continue
        reg = results[shape] - base
        if reg > NON_TARGET_SOFT_REGRESSION:
            score += 1.2 * reg
        if reg > NON_TARGET_HARD_REGRESSION:
            score += 6.0 * reg
    return score


def summarize_delta(results: dict):
    return {k: round(results[k] - v, 3) for k, v in BASELINES.items() if k in results}


def patch_cfg(cfg: dict, updates: dict):
    out = copy.deepcopy(cfg)
    out.update(updates)
    out["matrix_instr_nonkdim"] = 16
    return out


def dedup_trials(trials):
    seen = set()
    out = []
    for t in trials:
        key = (t["shape_idx"], json.dumps(t["trial_config"], sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def make_grid_trials(shape_idx: int, base_cfg: dict, axes: dict):
    import itertools
    label = SHAPES[shape_idx]["label"]
    file = SHAPES[shape_idx]["file"]
    tier = SHAPES[shape_idx]["tier"]
    keys = list(axes.keys())
    trials = []
    for values in itertools.product(*[axes[k] for k in keys]):
        upd = dict(zip(keys, values))
        if all(base_cfg.get(k) == v for k, v in upd.items()):
            continue
        cfg = patch_cfg(base_cfg, upd)
        trials.append({
            "shape_idx": shape_idx,
            "shape": label,
            "config_file": file,
            "tier": tier,
            "param": "+".join(keys),
            "value": "+".join(f"{k}={upd[k] if upd[k] is not None else 'null'}" for k in keys),
            "trial_config": cfg,
        })
    return trials


def gen_m256_trials(base_cfg: dict):
    trials = []
    # Core wide sweep: stay in v55 family, only resource/meta knobs.
    trials += make_grid_trials(5, base_cfg, {
        "num_warps": [4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4],
        "GROUP_SIZE_M": [2, 4, 8],
        "cache_modifier": [".cg", None],
    })
    # Small edge probes around the same family.
    singles = [
        {"BLOCK_SIZE_M": 16},
        {"BLOCK_SIZE_N": 256},
        {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "GROUP_SIZE_M": 4},
        {"BLOCK_SIZE_M": 16, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "GROUP_SIZE_M": 4},
        {"BLOCK_SIZE_N": 256, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
    ]
    for upd in singles:
        trials.append({
            "shape_idx": 5,
            "shape": SHAPES[5]["label"],
            "config_file": SHAPES[5]["file"],
            "tier": SHAPES[5]["tier"],
            "param": "+".join(sorted(upd.keys())),
            "value": "+".join(f"{k}={upd[k]}" for k in sorted(upd.keys())),
            "trial_config": patch_cfg(base_cfg, upd),
        })
    return dedup_trials(trials)


def gen_m64_trials(base_cfg: dict):
    trials = []
    trials += make_grid_trials(4, base_cfg, {
        "num_warps": [4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4],
        "GROUP_SIZE_M": [1, 4],
        "cache_modifier": [".cg", None],
    })
    extras = [
        {"BLOCK_SIZE_N": 256, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "GROUP_SIZE_M": 1},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
    ]
    for upd in extras:
        trials.append({
            "shape_idx": 4,
            "shape": SHAPES[4]["label"],
            "config_file": SHAPES[4]["file"],
            "tier": SHAPES[4]["tier"],
            "param": "+".join(sorted(upd.keys())),
            "value": "+".join(f"{k}={upd[k] if upd[k] is not None else 'null'}" for k in sorted(upd.keys())),
            "trial_config": patch_cfg(base_cfg, upd),
        })
    return dedup_trials(trials)


def gen_m16_trials(base_cfg: dict):
    label = SHAPES[1]["label"]
    file = SHAPES[1]["file"]
    tier = SHAPES[1]["tier"]
    trials = []
    curated = [
        {"num_warps": 4},
        {"num_stages": 1},
        {"waves_per_eu": 1},
        {"waves_per_eu": 4},
        {"cache_modifier": None},
        {"NUM_KSPLIT": 7},
        {"num_warps": 4, "num_stages": 2},
        {"num_warps": 8, "num_stages": 1},
        {"waves_per_eu": 1, "cache_modifier": None},
        {"waves_per_eu": 4, "cache_modifier": None},
        {"NUM_KSPLIT": 7, "cache_modifier": None},
        {"NUM_KSPLIT": 7, "waves_per_eu": 1},
    ]
    for upd in curated:
        trials.append({
            "shape_idx": 1,
            "shape": label,
            "config_file": file,
            "tier": tier,
            "param": "+".join(sorted(upd.keys())),
            "value": "+".join(f"{k}={upd[k] if upd[k] is not None else 'null'}" for k in sorted(upd.keys())),
            "trial_config": patch_cfg(base_cfg, upd),
        })
    return dedup_trials(trials)


def gen_k512_trials(shape_idx: int, base_cfg: dict):
    label = SHAPES[shape_idx]["label"]
    file = SHAPES[shape_idx]["file"]
    tier = SHAPES[shape_idx]["tier"]
    trials = []

    if shape_idx == 0:
        curated = [
            {"BLOCK_SIZE_N": 64},
            {"num_warps": 2},
            {"waves_per_eu": 1},
            {"waves_per_eu": 4},
            {"cache_modifier": None},
            {"BLOCK_SIZE_N": 64, "num_warps": 2},
        ]
    elif shape_idx == 2:
        curated = [
            {"num_warps": 4},
            {"waves_per_eu": 1},
            {"waves_per_eu": 4},
            {"cache_modifier": None},
            {"num_warps": 4, "waves_per_eu": 1},
            {"num_warps": 4, "cache_modifier": None},
        ]
    else:  # shape_idx == 3
        curated = [
            {"num_warps": 4},
            {"waves_per_eu": 1},
            {"waves_per_eu": 4},
            {"cache_modifier": None},
            {"num_warps": 4, "waves_per_eu": 1},
            {"num_warps": 4, "cache_modifier": None},
        ]

    for upd in curated:
        trials.append({
            "shape_idx": shape_idx,
            "shape": label,
            "config_file": file,
            "tier": tier,
            "param": "+".join(sorted(upd.keys())),
            "value": "+".join(f"{k}={upd[k] if upd[k] is not None else 'null'}" for k in sorted(upd.keys())),
            "trial_config": patch_cfg(base_cfg, upd),
        })
    return dedup_trials(trials)


def build_trials(configs: dict):
    trials = []
    trials.extend(gen_m256_trials(configs[SHAPES[5]["file"]][SHAPES[5]["tier"]]))
    trials.extend(gen_m64_trials(configs[SHAPES[4]["file"]][SHAPES[4]["tier"]]))
    trials.extend(gen_m16_trials(configs[SHAPES[1]["file"]][SHAPES[1]["tier"]]))
    trials.extend(gen_k512_trials(0, configs[SHAPES[0]["file"]][SHAPES[0]["tier"]]))
    trials.extend(gen_k512_trials(2, configs[SHAPES[2]["file"]][SHAPES[2]["tier"]]))
    trials.extend(gen_k512_trials(3, configs[SHAPES[3]["file"]][SHAPES[3]["tier"]]))
    trials = dedup_trials(trials)
    trials.sort(key=lambda t: (SHAPES[t["shape_idx"]]["priority"], t["shape"], t["param"], str(t["value"])))
    return trials


def analyze(log_paths):
    entries = []
    for p in log_paths:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    if not entries:
        print("No logs found")
        return

    entries.sort(key=lambda e: e.get("weighted_score", 1e9))
    print("=" * 150)
    print(f"Baseline geomean (v55): {baseline_geo_mean():.4f}")
    print(f"Entries: {len(entries)}")
    print("=" * 150)
    print(f"{'score':>8} | {'geo':>6} | {'shape':>18} | {'param':>28} | {'value':>42} | {'target':>6} | {'deltas'}")
    print("-" * 150)
    for e in entries[:30]:
        t = e["trial"]
        target = e["results"].get(t["shape"], 999) if e.get("results") else 999
        print(f"{e.get('weighted_score', 999):8.3f} | {e.get('geo_mean', 999):6.3f} | {t['shape']:>18} | {t['param'][:28]:>28} | {str(t['value'])[:42]:>42} | {target:6.2f} | {e.get('deltas', {})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--total-workers", type=int, default=3)
    ap.add_argument("--base", default=str(SCRIPT_DIR / "submission_v55.py"))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit after worker split")
    args = ap.parse_args()

    if args.analyze:
        logs = [SCRIPT_DIR / f"worker_{i}" / "search_log_v55_3w.jsonl" for i in range(1, 9)]
        analyze(logs)
        return

    if args.worker is None:
        ap.error("--worker required unless --analyze")

    worker_dir = SCRIPT_DIR / f"worker_{args.worker}"
    if not args.dry_run and not (worker_dir / ".popcorn.yaml").exists():
        print(f"ERROR: {worker_dir}/.popcorn.yaml not found")
        sys.exit(1)

    base_path = Path(args.base)
    configs, base_code = load_shape_configs(base_path)
    all_trials = build_trials(configs)
    my_trials = all_trials[args.worker - 1 :: args.total_workers]
    if args.limit > 0:
        my_trials = my_trials[: args.limit]

    print(f"Base: {base_path}")
    print(f"Worker {args.worker}/{args.total_workers}")
    print(f"Total candidate trials: {len(all_trials)} | This worker: {len(my_trials)}")
    print(f"Baseline geomean: {baseline_geo_mean():.4f}")

    if args.dry_run:
        for i, t in enumerate(my_trials, 1):
            print(f"[{i:03d}] {t['shape']} | {t['param']} | {t['value']}")
        return

    log_path = worker_dir / "search_log_v55_3w.jsonl"
    sub_path = SCRIPT_DIR / f"_sub_v55_3w_w{args.worker}.py"

    queue = [
        {"trial": trial, "orig_idx": idx, "pass_idx": 1, "defer_count": 0}
        for idx, trial in enumerate(my_trials, 1)
    ]
    deferred = []
    completed = 0
    total_planned = len(queue)
    max_defer_rounds = 2

    while queue:
        item = queue.pop(0)
        trial = item["trial"]
        i = item["orig_idx"]
        pass_idx = item["pass_idx"]

        print("\n" + "=" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trial {completed + 1}/{total_planned} (orig #{i}, pass {pass_idx})")
        print(f"  Target: {trial['shape']}")
        print(f"  Param : {trial['param']}")
        print(f"  Value : {trial['value']}")

        patched = copy.deepcopy(configs)
        patched[trial["config_file"]][trial["tier"]] = trial["trial_config"]
        sub_code = make_submission(base_code, patched)
        sub_path.write_text(sub_code)

        submit_info = submit(sub_path, worker_dir)
        raw = submit_info["raw"]
        status = submit_info["status"]
        attempts = submit_info["attempts"]
        results = parse_results(raw) if status == "ok" else {}
        geo = geo_mean(results)
        score = weighted_score(results, trial["shape"])
        deltas = summarize_delta(results)

        if status == "ok":
            completed += 1
            print(f"  attempts= {attempts}")
            print(f"  geo    = {geo:.4f}")
            print(f"  score  = {score:.4f}")
            print(f"  deltas = {deltas}")
        elif status == "defer" and item["defer_count"] < max_defer_rounds:
            item["defer_count"] += 1
            item["pass_idx"] += 1
            deferred.append(item)
            print(f"  transient error after {attempts} attempts; defer to end (defer_count={item['defer_count']})")
        else:
            completed += 1
            print(f"  status = {status}")
            print(f"  attempts= {attempts}")
            print(f"  No parsed benchmark results.")

        entry = {
            "ts": datetime.now().isoformat(),
            "worker": args.worker,
            "trial": {k: v for k, v in trial.items() if k != "trial_config"},
            "config_summary": {
                k: trial["trial_config"].get(k)
                for k in [
                    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                    "num_warps", "num_stages", "waves_per_eu", "cache_modifier",
                    "NUM_KSPLIT",
                ]
            },
            "submit_status": status,
            "submit_attempts": attempts,
            "pass_idx": pass_idx,
            "defer_count": item["defer_count"],
            "results": results,
            "geo_mean": geo,
            "weighted_score": score,
            "deltas": deltas,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        if status != "ok":
            err_path = worker_dir / f"error_v55_3w_{i}_pass{pass_idx}.txt"
            err_path.write_text(raw)
            print(f"  Raw output saved to {err_path}")

        if not queue and deferred:
            print(f"\n--- Starting deferred retry round with {len(deferred)} trials ---")
            queue, deferred = deferred, []

    print("\nDone. Analyze with: python3 auto_search1_v55_3workers.py --analyze")


if __name__ == "__main__":
    main()
