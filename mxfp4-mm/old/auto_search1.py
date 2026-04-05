#!/usr/bin/env python3
"""
MXFP4-MM automated parameter search.

12hr plan: 3 workers × 6/hr × 12hr = 216 submissions
  Worker 1: shape 4(M=64) → shape 0(M=4)   → combos on best
  Worker 2: shape 1(M=16) → shape 2(M=32a) → combos on best
  Worker 3: shape 5(M=256)→ shape 3(M=32b) → combos on best

Usage:
  python3 auto_search.py --worker 1 --shapes 4,0    # M=64 then M=4
  python3 auto_search.py --worker 2 --shapes 1,2    # M=16 then M=32/N=4096
  python3 auto_search.py --worker 3 --shapes 5,3    # M=256 then M=32/N=2880
  python3 auto_search.py --analyze
"""

import os, sys, json, time, re, copy, math, subprocess, argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Shapes
# ============================================================
SHAPES = {
    0: {"label": "M4_K512_N2880",    "M": 4,   "K": 512,  "N": 2880,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json",   "tier": "M_LEQ_8"},
    1: {"label": "M16_K7168_N2112",   "M": 16,  "K": 7168, "N": 2112,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json",  "tier": "M_LEQ_32"},
    2: {"label": "M32_K512_N4096",    "M": 32,  "K": 512,  "N": 4096,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=4096-K=512.json",   "tier": "M_LEQ_32"},
    3: {"label": "M32_K512_N2880",    "M": 32,  "K": 512,  "N": 2880,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json",   "tier": "M_LEQ_32"},
    4: {"label": "M64_K2048_N7168",   "M": 64,  "K": 2048, "N": 7168,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=7168-K=2048.json",  "tier": "M_LEQ_64"},
    5: {"label": "M256_K1536_N3072",  "M": 256, "K": 1536, "N": 3072,
        "file": "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=3072-K=1536.json",  "tier": "M_LEQ_256"},
}

# Best known configs (v38)
BEST_CONFIGS = {
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2880-K=512.json": {
        "M_LEQ_8":  {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "M_LEQ_32": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any":      {"BLOCK_SIZE_M": 32,"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=4096-K=512.json": {
        "M_LEQ_8":  {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "M_LEQ_32": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any":      {"BLOCK_SIZE_M": 32,"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 14},
        "any":      {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=7168-K=2048.json": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any":      {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1},
    },
    "gfx950-GEMM-A16WFP4_PRESHUFFLED-N=3072-K=1536.json": {
        "M_LEQ_64":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 3},
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
        "any":       {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg", "NUM_KSPLIT": 1},
    },
}

# ============================================================
# Trial generator
# ============================================================
def gen_trials(shape_idx):
    s = SHAPES[shape_idx]
    M, K, N = s["M"], s["K"], s["N"]
    K_half = K // 2
    base = copy.deepcopy(BEST_CONFIGS[s["file"]][s["tier"]])

    space = {
        "BLOCK_SIZE_M":  [v for v in [4, 8, 16, 32, 64] if v <= max(M, 4)],
        "BLOCK_SIZE_N":  [32, 64, 128, 256],
        "BLOCK_SIZE_K":  [v for v in [128, 256, 512] if v <= 2 * K_half],
        "num_warps":     [2, 4, 8],
        "num_stages":    [1, 2, 3],
        "waves_per_eu":  [0, 1, 2, 4],
        "GROUP_SIZE_M":  [1, 4, 8],
        "cache_modifier": [".cg", None],
        "NUM_KSPLIT":    [v for v in [1, 2, 3, 4, 7, 14] if K_half >= v * 64],
    }

    trials = []

    # Phase 1: coordinate descent (vary 1 param at a time)
    for param, values in space.items():
        for val in values:
            if val == base.get(param):
                continue
            cfg = copy.deepcopy(base)
            cfg[param] = val
            cfg["matrix_instr_nonkdim"] = 16
            trials.append({
                "shape": s["label"], "param": param,
                "value": val if val is not None else "null",
                "config_file": s["file"], "tier": s["tier"],
                "trial_config": cfg,
            })

    # Phase 2: promising 2-param combos (based on v37b/v38 findings)
    combos = [
        # BSK + stages (v38 showed BSK=256+stages=2 helps)
        ("BLOCK_SIZE_K", "num_stages"),
        # BSK + waves (lower BSK might need different waves)
        ("BLOCK_SIZE_K", "waves_per_eu"),
        # waves + warps (occupancy tuning)
        ("waves_per_eu", "num_warps"),
        # BSM + BSN (tile shape)
        ("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
    ]
    for p1, p2 in combos:
        for v1 in space.get(p1, []):
            for v2 in space.get(p2, []):
                if v1 == base.get(p1) and v2 == base.get(p2):
                    continue
                cfg = copy.deepcopy(base)
                cfg[p1] = v1
                cfg[p2] = v2
                cfg["matrix_instr_nonkdim"] = 16
                key = f"{p1}={v1 if v1 is not None else 'null'}+{p2}={v2 if v2 is not None else 'null'}"
                trials.append({
                    "shape": s["label"], "param": f"{p1}+{p2}",
                    "value": key,
                    "config_file": s["file"], "tier": s["tier"],
                    "trial_config": cfg,
                })

    # Deduplicate by config content
    seen = set()
    unique = []
    for t in trials:
        cfg_key = json.dumps(t["trial_config"], sort_keys=True, default=str)
        if cfg_key not in seen:
            seen.add(cfg_key)
            unique.append(t)

    return unique

# ============================================================
# Submission generator (patches SHAPE_CONFIGS in base file)
# ============================================================
def make_submission(base_path, configs):
    with open(base_path) as f:
        code = f.read()

    # Replace SHAPE_CONFIGS block
    start = code.index("SHAPE_CONFIGS = {")
    end = code.index("\n\n# ===", start)
    # json.dumps outputs null/true/false, convert to Python None/True/False
    cfg_str = json.dumps(configs, indent=4, default=str)
    cfg_str = cfg_str.replace(": null", ": None").replace(": true", ": True").replace(": false", ": False")
    new_block = "SHAPE_CONFIGS = " + cfg_str
    return code[:start] + new_block + code[end:]

# ============================================================
# Submit with HOME isolation + rate limit retry
# ============================================================
def submit(submission_path, worker_dir):
    env = os.environ.copy()
    env["HOME"] = os.path.abspath(worker_dir)

    cmd = [
        "popcorn-cli", "submit",
        "--mode", "benchmark",
        "--gpu", "MI355X",
        "--leaderboard", "amd-mxfp4-mm",
        os.path.abspath(submission_path),
        "--no-tui",
    ]

    while True:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Submitting (HOME={worker_dir})...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=700, env=env)
            output = r.stdout + "\n" + r.stderr
        except subprocess.TimeoutExpired:
            output = "TIMEOUT"
        except Exception as e:
            output = f"ERROR: {e}"

        # Check rate limit
        rate_match = re.search(r'Rate limit exceeded.*?Try again in (\d+)s', output)
        if rate_match:
            wait = int(rate_match.group(1)) + 10
            print(f"  Rate limited. Waiting {wait}s ({wait//60}m)...")
            time.sleep(wait)
            continue

        return output

# ============================================================
# Parse results
# ============================================================
def parse_results(output):
    results = {}
    for m in re.finditer(r'k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+).*?\n[^⏱]*⏱\s*([\d.]+)', output):
        k, mv, n, t = m.groups()
        results[f"M{mv}_K{k}_N{n}"] = float(t)
    return results

def geo_mean(results):
    if not results:
        return 999.0
    v = list(results.values())
    return math.exp(sum(math.log(x) for x in v) / len(v))

# ============================================================
# Analyze all worker logs
# ============================================================
def analyze():
    entries = []
    for log_path in sorted(Path(SCRIPT_DIR).glob("worker_*/search_log.jsonl")):
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

    if not entries:
        print("No results found in worker_*/search_log.jsonl")
        return

    entries.sort(key=lambda x: x.get("geo_mean", 999))

    print(f"\n{'='*80}")
    print(f"Total trials: {len(entries)}")
    print(f"{'='*80}")
    print(f"\nTop 15 by geo mean:")
    print(f"{'Geo':>7} | {'Shape':>22} | {'Param':>16} = {'Value':>6} | Per-shape times")
    print(f"{'-'*95}")

    for e in entries[:15]:
        t = e.get("trial", {})
        r = e.get("results", {})
        times = " ".join(f"{v:.1f}" for v in r.values()) if r else "N/A"
        print(f"{e.get('geo_mean',999):>6.2f} | {t.get('shape','?'):>22} | "
              f"{t.get('param','?'):>16} = {str(t.get('value','?')):>6} | {times}")

    # Best per shape
    print(f"\nBest config per target shape:")
    by_shape = {}
    for e in entries:
        shape = e.get("trial", {}).get("shape", "?")
        if shape not in by_shape or e.get("geo_mean", 999) < by_shape[shape]["geo_mean"]:
            by_shape[shape] = e
    for shape, e in sorted(by_shape.items()):
        t = e["trial"]
        cfg = e.get("config_summary", {})
        print(f"  {shape:>22}: geo={e['geo_mean']:.2f} | {t['param']}={t['value']}")
        if cfg:
            print(f"    config: {json.dumps(cfg)}")

# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, help="Worker ID (1-3)")
    p.add_argument("--total-workers", type=int, default=3)
    p.add_argument("--shapes", type=str, help="Shape indices, comma-separated (e.g. 4,0)")
    p.add_argument("--base", default=os.path.join(SCRIPT_DIR, "submission_v38.py"))
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.analyze:
        analyze()
        return

    if args.worker is None or args.shapes is None:
        p.error("Need --worker and --shapes (or --analyze)")

    shape_indices = [int(x) for x in args.shapes.split(",")]
    worker_dir = os.path.join(SCRIPT_DIR, f"worker_{args.worker}")
    log_file = os.path.join(worker_dir, "search_log.jsonl")

    # Gather all trials across shapes
    all_trials = []
    for si in shape_indices:
        trials = gen_trials(si)
        my_trials = trials[args.worker - 1::args.total_workers]
        all_trials.extend(my_trials)
        print(f"  Shape {si} ({SHAPES[si]['label']}): {len(my_trials)}/{len(trials)} trials")

    print(f"Worker {args.worker} | Total trials: {len(all_trials)} | Log: {log_file}")

    if args.dry_run:
        for i, t in enumerate(all_trials):
            print(f"  [{i+1}] {t['shape']} | {t['param']} = {t['value']}")
        est_hrs = len(all_trials) * 10.5 / 60  # ~10.5 min per trial
        print(f"\nEstimated time: {est_hrs:.1f} hours")
        return

    if args.dry_run:
        for i, t in enumerate(my_trials):
            print(f"  [{i+1}] {t['param']} = {t['value']}")
        return

    if not os.path.exists(args.base):
        print(f"ERROR: {args.base} not found"); sys.exit(1)
    if not os.path.exists(os.path.join(worker_dir, ".popcorn.yaml")):
        print(f"ERROR: {worker_dir}/.popcorn.yaml not found"); sys.exit(1)

    for i, trial in enumerate(all_trials):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[{ts}] Trial {i+1}/{len(all_trials)} | {trial['shape']} | {trial['param']} = {trial['value']}")

        # Build configs
        configs = copy.deepcopy(BEST_CONFIGS)
        configs[trial["config_file"]][trial["tier"]] = trial["trial_config"]

        # Generate submission
        try:
            sub_code = make_submission(args.base, configs)
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        sub_path = os.path.join(SCRIPT_DIR, f"_sub_w{args.worker}.py")
        with open(sub_path, 'w') as f:
            f.write(sub_code)

        # Submit (auto-retries on rate limit)
        output = submit(sub_path, worker_dir)
        results = parse_results(output)
        geo = geo_mean(results)

        target_time = results.get(trial["shape"], "N/A")
        print(f"  geo={geo:.2f} | target={target_time}")
        if results:
            print(f"  {' | '.join(f'{k}={v:.1f}' for k,v in results.items())}")

        # Log to worker's file
        entry = {
            "ts": datetime.now().isoformat(),
            "worker": args.worker,
            "trial": {k: v for k, v in trial.items() if k != "trial_config"},
            "config_summary": {
                k: trial["trial_config"][k]
                for k in ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K",
                           "num_warps","num_stages","waves_per_eu",
                           "GROUP_SIZE_M","NUM_KSPLIT","cache_modifier"]
            },
            "results": results,
            "geo_mean": geo,
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        # Failed submission: save raw output for debug
        if not results:
            err_path = os.path.join(worker_dir, f"error_{i}.txt")
            with open(err_path, 'w') as f:
                f.write(output)
            print(f"  No results! Raw output saved to {err_path}")

    print(f"\nDone! Run: python3 auto_search.py --analyze")

if __name__ == "__main__":
    main()
