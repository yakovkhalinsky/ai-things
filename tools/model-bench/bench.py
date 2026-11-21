#!/usr/bin/env python3
"""model-bench - compare local Ollama models on speed and quality.

The harness runs every model through:

  * a cold-load measurement (one tiny request after unloading the model),
  * a warm throughput benchmark at three prompt sizes, with a discarded
    warmup run and median-over-repeats aggregation,
  * a deterministic quality suite (tasks.py) with per-task graders.

and writes results.json, report.md and report.csv under results/<timestamp>/.
Console output is a compact summary table.

Stdlib only. Requires a reachable Ollama server and `ollama` on PATH for pulls.

Examples:
  ./bench.py --installed
  ./bench.py -m qwen3:4b -m llama3.2:3b --suite full --repeats 3
  ./bench.py -m qwen3:4b --speed-only --no-think
  ./bench.py --suggest
  ./bench.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import tasks as taskmod

HERE = Path(__file__).resolve().parent
SUGGESTIONS = HERE / "models.txt"

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA.startswith(("http://", "https://")):
    OLLAMA = "http://" + OLLAMA
OLLAMA = OLLAMA.rstrip("/")


# --- HTTP ------------------------------------------------------------------

def api(path: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        OLLAMA + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(f"{path}: HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA}: {exc.reason}") from None
    try:
        return json.loads(body or "{}")
    except json.JSONDecodeError:
        raise RuntimeError(f"{path}: non-JSON response: {body[:300]!r}") from None


def server_version() -> str:
    return api("/api/version", timeout=10).get("version", "?")


def installed_models() -> dict[str, dict]:
    return {m["name"]: m for m in api("/api/tags", timeout=30).get("models", [])}


def model_details(model: str) -> dict:
    try:
        info = api("/api/show", {"model": model}, timeout=60)
    except RuntimeError:
        return {}
    details = info.get("details") or {}
    mi = info.get("model_info") or {}
    arch = mi.get("general.architecture", "")
    ctx_candidates = [
        v for k, v in mi.items() if k.endswith(".context_length") and isinstance(v, (int, float))
    ]
    digest = None
    wanted = {model, model + ":latest"}
    try:
        for entry in api("/api/tags", timeout=30).get("models", []):
            if entry.get("name") in wanted:
                digest = entry.get("digest")
                break
    except RuntimeError:
        pass
    return {
        "digest": digest,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "format": details.get("format"),
        "max_ctx": int(max(ctx_candidates)) if ctx_candidates else None,
        "context_length": mi.get(f"{arch}.context_length"),
        "capabilities": info.get("capabilities") or [],
        "license": (info.get("license") or "").splitlines()[0][:40] if info.get("license") else None,
    }


def ps_entry(model: str) -> dict:
    try:
        running = api("/api/ps", timeout=15).get("models", [])
    except RuntimeError:
        return {}
    for entry in running:
        name = entry.get("name") or entry.get("model") or ""
        if name == model or name.split(":")[0] == model.split(":")[0]:
            return entry
    return {}


def unload(model: str) -> None:
    try:
        api("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except RuntimeError:
        pass


def pull(model: str) -> bool:
    if not shutil.which("ollama"):
        print(f"  ! cannot pull {model}: `ollama` not on PATH", file=sys.stderr)
        return False
    print(f"  pulling {model} ...", file=sys.stderr, flush=True)
    proc = subprocess.run(["ollama", "pull", model])
    return proc.returncode == 0


# --- generation ------------------------------------------------------------

def _payload(model, prompt, options, stream, system=None, fmt=None, think=None):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": options,
        "keep_alive": "10m",
    }
    if system:
        body["system"] = system
    if fmt:
        body["format"] = fmt
    if think is not None:
        body["think"] = think
    return body


def generate(model, prompt, options, timeout, system=None, fmt=None, think=None) -> dict:
    body = _payload(model, prompt, options, False, system, fmt, think)
    return api("/api/generate", body, timeout=timeout)


def timed_generate(model, prompt, options, timeout, think=None) -> dict:
    """Streaming request timed end-to-end; returns wall/TTFT/ollama counters."""
    body = _payload(model, prompt, options, True, think=think)
    req = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token: float | None = None
    parts: list[str] = []
    final: dict = {}
    done_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("response") or ""
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    parts.append(piece)
                if chunk.get("done"):
                    final = chunk
                    done_reason = chunk.get("done_reason")
                    break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"/api/generate: HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA}: {exc.reason}") from None

    end = time.perf_counter()
    if not final:
        raise RuntimeError("stream ended without a done chunk")

    def rate(count, duration_ns):
        return (count / (duration_ns / 1e9)) if count and duration_ns else None

    prompt_count = final.get("prompt_eval_count") or 0
    eval_count = final.get("eval_count") or 0
    result = {
        "wall_s": end - start,
        "ttft_ms": (first_token - start) * 1000 if first_token else None,
        "load_s": (final.get("load_duration") or 0) / 1e9,
        "prompt_tokens": prompt_count,
        "prompt_eval_s": (final.get("prompt_eval_duration") or 0) / 1e9,
        "prefill_tps": rate(prompt_count, final.get("prompt_eval_duration") or 0),
        "output_tokens": eval_count,
        "decode_s": (final.get("eval_duration") or 0) / 1e9,
        "decode_tps": rate(eval_count, final.get("eval_duration") or 0),
        "ollama_total_s": (final.get("total_duration") or 0) / 1e9,
        "done_reason": done_reason,
        "text": "".join(parts),
    }
    return result


# --- speed prompts ---------------------------------------------------------

PASSAGE = (
    "Packet switching is a method of grouping data that is transmitted over a "
    "digital network into packets. Packets are made of a header and a payload. "
    "Data in the header is used by networking hardware to direct the packet to its "
    "destination, where the payload is extracted and used by an operating system, "
    "application software, or higher layer protocols. Packet switching is the "
    "primary basis for data communications in computer networks worldwide. "
)

SPEED_CASES = {
    "short": (
        "Write two sentences about the ocean.",
        "One short prompt; measures warm latency and decode speed with minimal prefill."
    ),
    "medium": (
        "Read the passage below, then explain in your own words what packet "
        "switching is and why it matters.\n\n" + PASSAGE * 3,
        "Roughly 500 prompt tokens; typical chat-sized context."
    ),
    "long": (
        "Read the passage below and list the three most important ideas it "
        "contains. Answer in prose.\n\n" + PASSAGE * 12,
        "Roughly 2k prompt tokens; shows prefill scaling and long-context cost."
    ),
}


# --- measurement -----------------------------------------------------------

def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def cv(values):
    """Coefficient of variation in percent, or None with fewer than two values."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return (statistics.stdev(values) / mean * 100.0) if mean else None


def minmax(values):
    values = [v for v in values if v is not None]
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def bench_speed(model, args, think) -> dict:
    cases = [c.strip() for c in args.speed_cases.split(",") if c.strip() in SPEED_CASES]
    if not cases:
        raise SystemExit(f"--speed-cases must be a subset of {','.join(SPEED_CASES)}")
    options = {
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
    }
    runs: list[dict] = []
    if not args.no_warmup:
        print("    warmup ...", file=sys.stderr, flush=True)
        try:
            timed_generate(model, SPEED_CASES["short"][0], {**options, "num_predict": 16},
                           args.timeout, think=think)
        except RuntimeError as exc:
            print(f"    warmup failed: {exc}", file=sys.stderr)

    for case in cases:
        base = SPEED_CASES[case][0]
        for rep in range(args.repeats):
            # A unique tag per request defeats Ollama's prefix/KV cache; without
            # it, repeats of the same prompt report absurd prefill rates.
            prompt = f"(request {case}-{rep}-{secrets.token_hex(4)})\n{base}"
            run = timed_generate(model, prompt, options, args.timeout, think=think)
            run.update({"case": case, "rep": rep})
            runs.append(run)
            dec = f"{run['decode_tps']:.1f}" if run["decode_tps"] else "?"
            pre = f"{run['prefill_tps']:.0f}" if run["prefill_tps"] else "?"
            ttft = f"{run['ttft_ms']:.0f}" if run["ttft_ms"] else "?"
            print(f"    {case:6s} #{rep + 1}: decode {dec:>6s} tok/s | "
                  f"prefill {pre:>6s} tok/s | ttft {ttft:>5s} ms",
                  file=sys.stderr, flush=True)

    for run in runs:
        run.pop("text", None)
    return {
        "repeats": args.repeats,
        "cases": cases,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
        "runs": runs,
        "decode_tps": median([r["decode_tps"] for r in runs]),
        "prefill_tps": median([r["prefill_tps"] for r in runs]),
        "ttft_ms": median([r["ttft_ms"] for r in runs]),
        "wall_s": median([r["wall_s"] for r in runs]),
        "decode_range": minmax([r["decode_tps"] for r in runs]),
        "prefill_range": minmax([r["prefill_tps"] for r in runs]),
        "by_case": {
            case: {
                "decode_tps": median([r["decode_tps"] for r in runs if r["case"] == case]),
                "decode_cv": cv([r["decode_tps"] for r in runs if r["case"] == case]),
                "prefill_tps": median([r["prefill_tps"] for r in runs if r["case"] == case]),
                "prefill_cv": cv([r["prefill_tps"] for r in runs if r["case"] == case]),
                "ttft_ms": median([r["ttft_ms"] for r in runs if r["case"] == case]),
                "prompt_tokens": median([r["prompt_tokens"] for r in runs if r["case"] == case]),
            }
            for case in cases
        },
    }


def browser_selftest() -> tuple[bool, str]:
    """Confirm node + the optional Playwright smoke script can launch a browser."""
    node = shutil.which("node")
    script = HERE / "browser_smoke.mjs"
    if not node:
        return False, "node not found on PATH"
    if not script.exists():
        return False, f"missing {script.name}"
    try:
        proc = subprocess.run(
            [node, str(script), "--check"], cwd=str(HERE),
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "browser self-test timed out"
    if proc.returncode == 0:
        return True, "ok"
    lines = (proc.stderr or proc.stdout).strip().splitlines()
    return False, (lines[-1] if lines else f"exit {proc.returncode}")[:500]


def run_browser_check(task, response_path: Path, model_dir: Path, timeout: float) -> dict:
    """Run the optional Playwright smoke test on a generated webapp."""
    node = shutil.which("node")
    script = HERE / "browser_smoke.mjs"
    if not node or not script.exists():
        return {"status": "skipped", "reason": "node or browser_smoke.mjs not available"}
    env = os.environ.copy()
    env["BENCH_SHOTS"] = str(model_dir)
    try:
        proc = subprocess.run(
            [node, str(script), str(response_path)], cwd=str(HERE),
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": f"timed out after {timeout:g}s"}
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return {"status": "error", "reason": (lines[-1] if lines else f"exit {proc.returncode}")[:400]}
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "error", "reason": "could not parse browser output"}
    item = data[0] if isinstance(data, list) and data else data
    if not isinstance(item, dict):
        return {"status": "error", "reason": "unexpected browser output"}
    return {"status": "ok", "result": item}


def browser_check_ok(check: str, result: dict) -> bool:
    if check == "canvas":
        return bool(result.get("canvas"))
    if check == "animating":
        return bool(result.get("animating"))
    if check == "no_errors":
        return not result.get("pageErrors") and not result.get("consoleErrors")
    if check == "game_over":
        return bool(result.get("gameOver"))
    if check == "restart":
        return bool(result.get("restarted"))
    return True


def browser_failures(checks: list[str], result: dict) -> list[str]:
    fails = []
    if not result:
        return ["no browser result"]
    if "canvas" in checks and not result.get("canvas"):
        fails.append("no canvas")
    if "animating" in checks and not result.get("animating"):
        fails.append("not animating")
    if "no_errors" in checks:
        for key, label in (("pageErrors", "page error"), ("consoleErrors", "console error")):
            if result.get(key):
                fails.append(f"{label}: {result[key][0]}")
    if "game_over" in checks and not result.get("gameOver"):
        fails.append("no game over")
    if "restart" in checks and not result.get("restarted"):
        fails.append("restart failed")
    return fails


def bench_quality(model, args, suite_tasks, think, response_dir: Path) -> dict:
    records = []
    model_dir = response_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    model_dir.mkdir(parents=True, exist_ok=True)

    for task in suite_tasks:
        print(f"    {task['id']:<18s} ...", end="", file=sys.stderr, flush=True)
        options = {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": task.get("num_predict", 300),
            "num_ctx": task.get("num_ctx", args.num_ctx),
        }
        timeout = task.get("timeout_s", args.timeout)
        started = time.perf_counter()
        browser = None
        try:
            resp = generate(model, task["prompt"], options, timeout,
                            fmt=task.get("format"), think=think)
            text = resp.get("response") or ""
            done_reason = resp.get("done_reason")
            repetition = taskmod.repetition_ratio(text)
            truncated = done_reason == "length"
            output_tokens = resp.get("eval_count")

            check_path = model_dir / f"{task['id']}.txt"
            if task.get("kind") == "webapp":
                check_path = model_dir / f"{task['id']}.html"
                check_path.write_text(taskmod.extract_html(text), encoding="utf-8")

            verdict = task["grader"](text)
            passed, detail = verdict[0], verdict[1]
            partial = float(verdict[2]) if len(verdict) > 2 else (1.0 if passed else 0.0)
            if truncated:
                passed = False
                partial = 0.0
                detail = f"truncated at {output_tokens or '?'} tokens; {detail}"

            if task.get("kind") == "webapp" and args.browser and not truncated:
                checks = task.get("browser_checks", [])
                browser = run_browser_check(task, check_path, model_dir, timeout)
                if browser["status"] == "ok":
                    if not browser_check_ok("no_errors", browser["result"]):
                        # A page that throws is not a partially working game.
                        browser_fraction = 0.0
                    elif checks:
                        browser_fraction = sum(
                            1 for check in checks if browser_check_ok(check, browser["result"])
                        ) / len(checks)
                    else:
                        browser_fraction = 1.0
                    # Runtime behaviour is the point of a webapp task, so it
                    # carries more weight than the static shape of the file.
                    partial = 0.4 * partial + 0.6 * browser_fraction
                    fails = browser_failures(checks, browser["result"])
                    if fails:
                        passed = False
                        detail = (detail + "; " if detail != "ok" else "") + "browser: " + "; ".join(fails)
                    elif detail == "ok":
                        detail = "ok; browser ok"
                else:
                    passed = False
                    partial *= 0.5
                    detail = ((detail + "; ") if detail != "ok" else "") + \
                        f"browser {browser['status']}: {browser.get('reason', '')}"

            record = {
                "id": task["id"],
                "category": task["category"],
                "weight": task.get("weight", 1.0),
                "passed": bool(passed),
                "score": partial,
                "detail": detail,
                "latency_s": time.perf_counter() - started,
                "prompt_tokens": resp.get("prompt_eval_count"),
                "output_tokens": output_tokens,
                "output_bytes": len(text.encode("utf-8")),
                "repetition": round(repetition, 3),
                "done_reason": done_reason,
                "truncated": truncated,
                "browser": browser,
                "output": text,
            }
        except Exception as exc:  # noqa: BLE001 - keep the suite running
            record = {
                "id": task["id"],
                "category": task["category"],
                "weight": task.get("weight", 1.0),
                "passed": False,
                "score": 0.0,
                "detail": f"error: {exc}",
                "latency_s": time.perf_counter() - started,
                "prompt_tokens": None,
                "output_tokens": None,
                "output_bytes": 0,
                "repetition": 0.0,
                "done_reason": None,
                "truncated": False,
                "browser": None,
                "output": "",
            }
        (model_dir / f"{task['id']}.txt").write_text(record["output"], encoding="utf-8")
        records.append(record)
        flags = " [truncated]" if record["truncated"] else ""
        if record.get("repetition", 0.0) >= 0.5:
            flags += f" [rep {100 * record['repetition']:.0f}%]"
        if record["browser"]:
            flags += f" [browser {record['browser']['status']}]"
        print(f" {'PASS' if record['passed'] else 'FAIL'}{flags}  {record['detail']}",
              file=sys.stderr, flush=True)

    total_weight = sum(r["weight"] for r in records) or 1.0
    score = sum(r["weight"] * r.get("score", 1.0 if r["passed"] else 0.0) for r in records) / total_weight
    categories: dict[str, dict] = {}
    for rec in records:
        cat = categories.setdefault(rec["category"], {"passed": 0, "total": 0, "weight": 0.0, "passed_weight": 0.0})
        cat["total"] += 1
        cat["weight"] += rec["weight"]
        cat["passed_weight"] += rec["weight"] * rec.get("score", 1.0 if rec["passed"] else 0.0)
        if rec["passed"]:
            cat["passed"] += 1
    for cat in categories.values():
        cat["score"] = cat["passed_weight"] / cat["weight"] if cat["weight"] else 0.0
    return {"score": score, "records": records, "categories": categories}


# --- reporting -------------------------------------------------------------

def fmt(value, digits=1, dash="-"):
    if value is None:
        return dash
    return f"{value:.{digits}f}"


def composite(results, args):
    speeds = [r["speed"]["decode_tps"] for r in results if r.get("speed")]
    best = max((s for s in speeds if s), default=0)
    for r in results:
        if r.get("error"):
            r["composite"] = 0.0
            r["speed_norm"] = 0.0
            continue
        quality = r.get("quality", {}).get("score", 0.0)
        decode = (r.get("speed") or {}).get("decode_tps") or 0
        speed_norm = (decode / best) if best else 0.0
        if args.speed_only:
            score = speed_norm
        elif args.quality_only:
            score = quality
        else:
            score = args.w_quality * quality + args.w_speed * speed_norm
        r["composite"] = score
        r["speed_norm"] = speed_norm
    return best


def mark_pareto(results) -> None:
    for r in results:
        if r.get("error"):
            r["pareto"] = False
            continue
        r["pareto"] = True
        q = r.get("quality", {}).get("score", 0.0)
        s = (r.get("speed") or {}).get("decode_tps") or 0
        for other in results:
            if other is r or other.get("error"):
                continue
            oq = other.get("quality", {}).get("score", 0.0)
            os_ = (other.get("speed") or {}).get("decode_tps") or 0
            if oq >= q and os_ >= s and (oq > q or os_ > s):
                r["pareto"] = False
                break


def console_table(headers, rows, aligns=None):
    aligns = aligns or ["<"] * len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(f"{str(h):{aligns[i]}{widths[i]}}" for i, h in enumerate(headers))
    out = [line, "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(row)))
    return "\n".join(out)


def md_table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def summary_rows(results, args):
    rows = []
    for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
        if r.get("error"):
            rows.append([r["model"], "ERROR"] + ["-"] * 11)
            continue
        speed = r.get("speed") or {}
        quality = r.get("quality") or {}
        meta = r.get("meta") or {}
        quality_cell = "n/a" if args.speed_only else f"{fmt(100 * quality.get('score', 0), 0)}%"
        truncated = sum(1 for rec in quality.get("records", []) if rec.get("truncated"))
        rows.append([
            r["model"],
            quality_cell,
            fmt(speed.get("decode_tps")),
            fmt(speed.get("prefill_tps"), 0),
            fmt(speed.get("ttft_ms"), 0),
            fmt(r.get("load_s"), 2),
            fmt(r.get("vram_mib"), 0),
            fmt(r.get("gpu_pct"), 0, "-"),
            str(truncated) if not args.speed_only else "-",
            fmt(100 * r.get("composite", 0), 1),
            "yes" if r.get("pareto") else "",
            meta.get("parameter_size") or "?",
            meta.get("quantization") or "?",
        ])
    return rows


SUMMARY_HEADERS = ["model", "quality", "decode t/s", "prefill t/s", "ttft ms",
                   "load s", "vram MiB", "gpu %", "trunc", "score", "pareto", "params", "quant"]


def build_report(results, args, version, host, suite_tasks, warnings) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    extra = []
    if args.category:
        extra.append("categories=" + ",".join(args.category))
    if args.task:
        extra.append("tasks=" + ",".join(args.task))
    extra.append("browser=" + ("on" if args.browser else "off"))
    lines = [
        "# model-bench report",
        "",
        f"- run: {stamp}",
        f"- host: {host}",
        f"- ollama: {version}  ({OLLAMA})",
        f"- tasks: {len(suite_tasks)} ({'full' if (args.category or args.task) else args.suite}"
        f"{' filtered' if (args.category or args.task) else ''}), "
        f"speed cases: {args.speed_cases}, repeats: {args.repeats}",
        f"- speed num_ctx: {args.num_ctx}, speed num_predict: {args.num_predict}, "
        f"temperature: {args.temperature}, seed: {args.seed}, think: "
        f"{'off' if args.no_think else 'model default'}",
        f"- composite weights: quality {args.w_quality:g}, speed {args.w_speed:g}",
        f"- filters: {'; '.join(extra)}",
        "",
        "## Summary",
        "",
        md_table(SUMMARY_HEADERS, summary_rows(results, args),
                 ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
                  "---:", ":-:", "---", "---"]),
        "",
        "`score` is the weighted quality score plus decode throughput normalised "
        "to the fastest model in this run. `pareto` marks models that no other "
        "model beats on both quality and speed. Speed is the median of warm runs "
        "after a discarded warmup; `load s` is the cold load of the first request. "
        "`trunc` counts tasks that hit their token budget; those are always failed.",
        "",
    ]

    model_rows = []
    for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
        meta = r.get("meta") or {}
        digest = meta.get("digest") or "?"
        if len(digest) > 23:
            digest = digest[:23] + "…"
        model_rows.append([
            r["model"],
            digest,
            meta.get("parameter_size") or "?",
            meta.get("quantization") or "?",
            meta.get("max_ctx") or "?",
        ])
    lines += ["## Models", "",
              md_table(["model", "digest", "params", "quant", "max ctx"],
                       model_rows, ["---", "---", "---", "---", "---:"]), ""]

    if not args.speed_only:
        cats = sorted({c for r in results for c in (r.get("quality") or {}).get("categories", {})})
        rows = []
        for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
            quality = r.get("quality") or {}
            cats_data = quality.get("categories", {})
            row = [r["model"], f"{fmt(100 * quality.get('score', 0), 0)}%"]
            for cat in cats:
                c = cats_data.get(cat)
                row.append(f"{c['passed']}/{c['total']} ({100 * c['score']:.0f}%)" if c else "-")
            rows.append(row)
        lines += ["## Quality by category", "",
                  md_table(["model", "total"] + cats, rows,
                           ["---", "---:"] + ["---:"] * len(cats)), ""]

        cost_rows = []
        for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
            recs = (r.get("quality") or {}).get("records", [])
            if not recs:
                continue
            max_rep = max((rec.get("repetition") or 0.0 for rec in recs), default=0.0)
            cost_rows.append([
                r["model"],
                len(recs),
                sum(rec.get("output_tokens") or 0 for rec in recs),
                fmt(sum(rec.get("latency_s") or 0 for rec in recs), 1),
                sum(rec.get("output_bytes") or 0 for rec in recs),
                sum(1 for rec in recs if rec.get("truncated")),
                f"{100 * max_rep:.0f}%",
            ])
        lines += ["## Quality cost", "",
                  md_table(["model", "tasks", "out tokens", "gen s", "out bytes", "trunc", "max rep"],
                           cost_rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
                  "",
                  "`max rep` is the highest 8-gram repetition ratio among a model's quality "
                  "outputs; high values usually mean a loop. It is diagnostic, not a pass/fail "
                  "rule.", ""]

    if not args.quality_only:
        cases = [c.strip() for c in args.speed_cases.split(",") if c.strip() in SPEED_CASES]
        rows = []
        for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
            speed = r.get("speed") or {}
            by_case = speed.get("by_case", {})
            row = [r["model"]]
            for case in cases:
                c = by_case.get(case, {})
                row.append(f"{fmt(c.get('decode_tps'))} / {fmt(c.get('prefill_tps'), 0)}")
            rows.append(row)
        lines += [
            "## Speed detail (decode t/s / prefill t/s)",
            "",
            md_table(["model"] + cases, rows, ["---"] + ["---:"] * len(cases)),
            "",
        ]
        for case in cases:
            lines.append(f"- {case}: {SPEED_CASES[case][1]}")
        lines.append("")

        def cv_range(values):
            values = [v for v in values if v is not None]
            if not values:
                return "-"
            if len(values) == 1:
                return f"{values[0]:.1f}%"
            return f"{min(values):.1f}–{max(values):.1f}%"

        lines += ["Within-case variance across repeats (coefficient of variation):", ""]
        for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
            by_case = (r.get("speed") or {}).get("by_case", {})
            decode_cvs = [c.get("decode_cv") for c in by_case.values() if c.get("decode_cv") is not None]
            prefill_cvs = [c.get("prefill_cv") for c in by_case.values() if c.get("prefill_cv") is not None]
            if decode_cvs or prefill_cvs:
                lines.append(f"- {r['model']}: decode {cv_range(decode_cvs)}, "
                             f"prefill {cv_range(prefill_cvs)}")
        lines.append("")

    failures = []
    for r in results:
        for rec in (r.get("quality") or {}).get("records", []):
            if not rec["passed"]:
                failures.append((r["model"], rec["id"], rec["category"], rec["detail"]))
    if failures and not args.speed_only:
        lines += ["## Failed tasks", "",
                  md_table(["model", "task", "category", "detail"], failures,
                           ["---", "---", "---", "---"]), ""]

    failed_models = [(r["model"], r["error"]) for r in results if r.get("error")]
    if failed_models:
        lines += ["## Failed models", "",
                  md_table(["model", "error"], failed_models, ["---", "---"]), ""]

    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    lines += [
        "## Reproduce",
        "",
        "```",
        " ".join([str(Path(__file__).name)] + args.argv),
        "```",
        "",
        f"Raw model outputs: `responses/` in this directory.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(out_dir: Path, results, args, version, host, suite_tasks, warnings) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path = out_dir / "report.md"
    csv_path = out_dir / "report.csv"

    payload = {
        "run": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "host": host,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ollama": version,
            "url": OLLAMA,
            "ollama_env": {
                key: os.environ[key]
                for key in (
                    "OLLAMA_HOST", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
                    "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE",
                    "OLLAMA_GPU_OVERHEAD", "OLLAMA_KEEP_ALIVE",
                )
                if key in os.environ
            },
            "argv": args.argv,
            "suite": args.suite,
            "speed_cases": args.speed_cases,
            "repeats": args.repeats,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
            "seed": args.seed,
            "no_think": args.no_think,
            "weights": {"quality": args.w_quality, "speed": args.w_speed},
            "labels": [t["id"] for t in suite_tasks],
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(build_report(results, args, version, host, suite_tasks, warnings), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "quality", "decode_tps", "prefill_tps", "ttft_ms",
                         "load_s", "vram_mib", "gpu_pct", "trunc", "composite", "pareto",
                         "params", "quant", "suite", "num_ctx", "digest"])
        for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
            if r.get("error"):
                writer.writerow([r["model"], "ERROR"] + [""] * 14)
                continue
            speed = r.get("speed") or {}
            quality = r.get("quality") or {}
            meta = r.get("meta") or {}
            truncated = sum(1 for rec in quality.get("records", []) if rec.get("truncated"))
            writer.writerow([
                r["model"], "" if args.speed_only else f"{quality.get('score', 0):.3f}", fmt(speed.get("decode_tps")),
                fmt(speed.get("prefill_tps"), 0), fmt(speed.get("ttft_ms"), 0),
                fmt(r.get("load_s"), 2), fmt(r.get("vram_mib"), 0),
                fmt(r.get("gpu_pct"), 0, ""), truncated,
                f"{r.get('composite', 0):.3f}",
                "yes" if r.get("pareto") else "", meta.get("parameter_size") or "",
                meta.get("quantization") or "", args.suite, args.num_ctx,
                meta.get("digest") or "",
            ])
    return json_path, md_path, csv_path


# --- per-model run ---------------------------------------------------------

def run_one(model: str, args, suite_tasks, response_dir: Path) -> dict:
    think = False if args.no_think else None
    result = {"model": model, "error": None}
    print(f"\n== {model}", file=sys.stderr, flush=True)

    meta = model_details(model)
    result["meta"] = meta
    if meta:
        print(f"   {meta.get('parameter_size') or '?'} params, "
              f"{meta.get('quantization') or '?'}, "
              f"max ctx {meta.get('max_ctx') or '?'}", file=sys.stderr)

    # Cold load: unload, then time one tiny request.
    print("    cold load ...", file=sys.stderr, flush=True)
    unload(model)
    time.sleep(0.5)
    try:
        cold = generate(model, "Say hi in two words.",
                        {"num_predict": 8, "temperature": 0, "num_ctx": args.num_ctx},
                        args.timeout, think=think)
        result["load_s"] = (cold.get("load_duration") or 0) / 1e9
        result["cold_wall_s"] = (cold.get("total_duration") or 0) / 1e9
        print(f"    loaded in {fmt(result['load_s'], 2)} s", file=sys.stderr, flush=True)
    except RuntimeError as exc:
        result["error"] = f"cold load failed: {exc}"
        print(f"    ! {result['error']}", file=sys.stderr)
        return result

    if not args.quality_only:
        try:
            result["speed"] = bench_speed(model, args, think)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"speed failed: {exc}"
            print(f"    ! {result['error']}", file=sys.stderr)
            return result

    if not args.speed_only:
        try:
            result["quality"] = bench_quality(model, args, suite_tasks, think, response_dir)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"quality failed: {exc}"
            print(f"    ! {result['error']}", file=sys.stderr)
            return result

    entry = ps_entry(model)
    if entry:
        size = entry.get("size") or 0
        vram = entry.get("size_vram") or 0
        result["size_bytes"] = size
        result["vram_bytes"] = vram
        result["vram_mib"] = vram / (1024 * 1024) if vram else None
        result["gpu_pct"] = (100 * vram / size) if size else None
        result["processor"] = entry.get("processor")
        result["context"] = entry.get("context")

    # Free VRAM before the next model is measured.
    unload(model)
    return result


# --- self test -------------------------------------------------------------

def self_test() -> int:
    failures = 0
    print(f"{'task':<20s} {'reference':<10s} {'empty':<7s} {'garbage':<8s} detail")
    for task in taskmod.TASKS:
        try:
            variants = [task["reference"]]
            if task.get("reference_alt"):
                variants.append(task["reference_alt"])
            ref_ok, ref_detail = True, "ok"
            for index, sample in enumerate(variants):
                verdict = task["grader"](sample)
                sample_ok, sample_detail = verdict[0], verdict[1]
                score = verdict[2] if len(verdict) > 2 else (1.0 if sample_ok else 0.0)
                if not 0.0 <= float(score) <= 1.0:
                    sample_ok, sample_detail = False, f"score {score!r} outside [0,1]"
                if not sample_ok:
                    ref_ok = False
                    ref_detail = f"variant {index}: {sample_detail}"
                    break
        except Exception as exc:  # noqa: BLE001
            ref_ok, ref_detail = False, f"raised {exc!r}"
        try:
            empty_ok = bool(task["grader"]("")[0])
        except Exception:  # noqa: BLE001
            empty_ok = False
        try:
            junk = task["grader"]("I'm sorry, but I cannot help with that.")
            junk_ok, junk_detail = bool(junk[0]), junk[1]
        except Exception:  # noqa: BLE001
            junk_ok = False
            junk_detail = "raised"
        bad = (not ref_ok) or empty_ok or junk_ok
        failures += 1 if bad else 0
        print(f"{task['id']:<20s} {'ok' if ref_ok else 'FAIL':<10s} "
              f"{'ok' if not empty_ok else 'FAIL':<7s} "
              f"{'ok' if not junk_ok else 'FAIL':<8s} "
              f"{ref_detail if not ref_ok else (junk_detail if junk_ok else '')}")
    print()
    if failures:
        print(f"{failures} task(s) failed the self-test", file=sys.stderr)
        return 1
    print(f"all {len(taskmod.TASKS)} tasks passed the self-test")
    return 0


# --- main ------------------------------------------------------------------

def host_summary() -> str:
    parts = [platform.node() or "?"]
    try:
        cpus = os.cpu_count()
        mem = None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem = int(line.split()[1]) / (1024 * 1024)
                    break
        parts.append(f"{cpus} vCPU")
        if mem:
            parts.append(f"{mem:.0f} GiB RAM")
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            parts.append(out)
            header = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5,
            ).stdout
            match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", header)
            if match:
                parts.append(f"CUDA {match.group(1)}")
        except (OSError, IndexError, subprocess.SubprocessError):
            pass
    return " | ".join(parts)


def parse_models(values: list[str]) -> list[str]:
    models: list[str] = []
    for value in values:
        for name in re.split(r"[,\s]+", value):
            if name and name not in models:
                models.append(name)
    return models


def suggestions() -> list[str]:
    if not SUGGESTIONS.exists():
        return []
    return [ln.strip() for ln in SUGGESTIONS.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare local Ollama models on speed and quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[-1],
    )
    parser.add_argument("-m", "--models", action="append", default=[],
                        help="model name; repeatable, or comma/space separated")
    parser.add_argument("--installed", action="store_true", help="bench every locally installed model")
    parser.add_argument("--suggest", action="store_true", help="print suggested models for this GPU class")
    parser.add_argument("--pull", action="store_true", help="pull missing models before benching")
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("-c", "--category", action="append", default=[],
                        help="only these task categories (repeatable, comma/space separated)")
    parser.add_argument("--task", action="append", default=[],
                        help="only these task ids (repeatable, comma/space separated)")
    parser.add_argument("--browser", action="store_true",
                        help="run headless browser checks for webapp tasks (node + playwright)")
    parser.add_argument("--speed-only", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--speed-cases", default="short,medium,long",
                        help="comma list from short,medium,long")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-predict", type=int, default=192)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-think", action="store_true",
                        help="send think=false (removes hidden reasoning tokens)")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--w-quality", type=float, default=0.6)
    parser.add_argument("--w-speed", type=float, default=0.4)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--label", default="")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list-tasks", action="store_true",
                        help="list quality tasks, categories, and suites")
    parser.add_argument("--list-installed", action="store_true")
    args = parser.parse_args(argv)
    args.argv = argv

    if args.list_tasks:
        print(f"{'id':<20} {'category':<12} {'suite':<6} {'weight':<6} note")
        for task in taskmod.TASKS:
            print(f"{task['id']:<20} {task['category']:<12} "
                  f"{'quick' if task['quick'] else 'full':<6} "
                  f"{task.get('weight', 1.0):<6} {task.get('note', '')}")
        return 0

    if args.self_test:
        return self_test()

    if args.browser:
        ok, message = browser_selftest()
        if not ok:
            print(f"error: --browser requested but unavailable: {message}", file=sys.stderr)
            print("  install node and run `npm install` (plus `npx playwright install chromium`)\n"
                  "  in tools/model-bench, or omit --browser for static checks only.", file=sys.stderr)
            return 2

    if args.suggest:
        print("# Suggested local models for ~8 GB VRAM (edit models.txt to taste)\n")
        for name in suggestions():
            print(name)
        return 0

    try:
        version = server_version()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    installed = installed_models()

    if args.list_installed:
        if not installed:
            print("no models installed")
        for name, meta in installed.items():
            print(f"{name:<40s} {meta.get('size', 0) / 1e9:6.2f} GB  "
                  f"{meta.get('details', {}).get('parameter_size', '?')}")
        return 0

    models = parse_models(args.models)
    if args.installed:
        models += [name for name in installed if name not in models]
    if not models:
        print("error: no models given. Use -m MODEL, --installed, or --suggest.",
              file=sys.stderr)
        return 2

    for name in models:
        if name.endswith(":cloud"):
            print(f"note: {name} runs in the cloud; its speed is network-bound",
                  file=sys.stderr)

    missing = [m for m in models if m not in installed and not m.endswith(":cloud")]
    if missing:
        if args.pull:
            missing = [m for m in missing if not pull(m)]
        else:
            for name in missing:
                print(f"error: {name} is not installed (use --pull or `ollama pull {name}`)",
                      file=sys.stderr)
            return 2
        still = [m for m in missing if m not in installed_models()]
        if still:
            print(f"error: could not pull {', '.join(still)}", file=sys.stderr)
            return 2

    filtered = bool(args.category or args.task)
    suite_tasks = taskmod.selected("full" if filtered else args.suite)
    if args.category:
        wanted = set(parse_models(args.category))
        suite_tasks = [t for t in suite_tasks if t["category"] in wanted]
    if args.task:
        wanted = set(parse_models(args.task))
        suite_tasks = [t for t in suite_tasks if t["id"] in wanted]
    if not suite_tasks:
        print("error: no tasks match the category/task filter", file=sys.stderr)
        return 2
    if args.browser and not any(t.get("kind") == "webapp" for t in suite_tasks):
        print("note: --browser only applies to webapp tasks; none selected", file=sys.stderr)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = f"-{args.label}" if args.label else ""
    out_dir = Path(args.out) / f"{stamp}{label}"
    response_dir = out_dir / "responses"
    host = host_summary()

    print(f"model-bench {version} | {host}", file=sys.stderr)
    selection = "full" if filtered else args.suite
    print(f"tasks={len(suite_tasks)} ({selection}{' filtered' if filtered else ''}) | "
          f"browser={'on' if args.browser else 'off'} | models={len(models)} | out={out_dir}",
          file=sys.stderr)

    results: list[dict] = []
    warnings: list[str] = []
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model}", file=sys.stderr, flush=True)
        try:
            results.append(run_one(model, args, suite_tasks, response_dir))
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001
            results.append({"model": model, "error": str(exc)})
            print(f"  ! {model}: {exc}", file=sys.stderr)

    for r in results:
        if r.get("error"):
            warnings.append(f"{r['model']}: {r['error']}")

    if not results:
        print("error: no model completed", file=sys.stderr)
        return 1

    composite(results, args)
    mark_pareto(results)
    json_path, md_path, csv_path = write_outputs(
        out_dir, results, args, version, host, suite_tasks, warnings
    )

    rows = summary_rows(results, args)
    print("\n" + console_table(SUMMARY_HEADERS, rows,
                               ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">",
                                "^", "<", "<"]))
    print(f"\nwrote {md_path}\n      {json_path}\n      {csv_path}", file=sys.stderr)
    if warnings:
        print("\nwarnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
