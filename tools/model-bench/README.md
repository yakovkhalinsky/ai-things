# model-bench

A small, dependency-free harness for trialling **local Ollama models** on this
machine: how fast they are, how good they are, and which one is worth using.

Everything is measured against the same prompts, the same context size and the
same deterministic graders, so the results are comparable within a run (and
roughly comparable across runs on the same GPU).

```
speed  ->  cold load time, time-to-first-token, prefill tok/s, decode tok/s
quality ->  15 deterministic tasks: JSON, instruction following, code (unit
            tests), reasoning, extraction, summarisation, SQL (executed),
            and a whole-app webapp task (single-file Snake)
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- model-bench
```

This puts `model-bench` in `~/bin`; install `ol` separately if you want it.
The rest of this README uses `./bench.py`; from an installed copy use
`model-bench` instead.

## Requirements

- Ollama reachable (default `http://localhost:11434`, honours `OLLAMA_HOST`)
- Python 3.10+ (stdlib only - no venv, no pip install)
- Optional: `nvidia-smi` for the GPU line in the report, `ollama` on PATH for `--pull`

Verified on this box: RTX 4060 Laptop 8 GB, 31 GiB RAM, WSL2, Ollama 0.40.

## Quick start

```bash
cd tools/model-bench

# sanity-check the graders themselves (no models needed)
./bench.py --self-test

# see candidates that suit ~8 GB VRAM
./bench.py --suggest

# bench two models, quick suite, 3 repeats each
./bench.py -m qwen3:4b -m llama3.2:3b

# pull missing models first
./bench.py -m qwen2.5-coder:7b --pull

# everything already installed
./bench.py --installed --suite full

# throughput only, no quality tasks, thinking models with hidden tokens off
./bench.py -m qwen3:8b --speed-only --no-think

# one task category; --browser adds headless runtime checks for webapp tasks
./bench.py -m qwen2.5-coder:7b --quality-only --category webapp --browser
```

Results land in `results/<UTC timestamp>/`:

| file | contents |
| --- | --- |
| `report.md` | summary, quality by category, speed detail, failures |
| `results.json` | full run config and per-run/per-task records |
| `report.csv` | one row per model, ready for a spreadsheet |
| `responses/<model>/<task>.txt` | raw model output, for inspecting failures |
| `responses/<model>/webapp_*.html` | extracted webapp, plus browser screenshots when `--browser` is on |

The console prints the same summary table:

```
model         quality  decode t/s  prefill t/s  ttft ms  load s  vram MiB  gpu %  trunc  score  pareto
------------  -------  ----------  -----------  -------  ------  --------  -----  -----  -----  ------
llama3.2:3b        92%        71.2         4100       45    2.10      2900    100      0   88.4   yes
qwen3:4b           96%        55.8         3200       63    2.60      3600    100      0   85.1   yes
qwen2.5:1.5b       71%       142.5         7800       22    1.40      1500    100      2   80.6
```

## What the numbers mean

**Speed** is sampled per prompt size (`short`, `medium`, `long`) with
`--repeats` runs each (default 3), after one discarded warmup request. Each
request carries a unique tag, because Ollama otherwise serves repeated identical
prompts from its prefix cache and reports meaningless prefill rates. The table
shows the **median** warm run. `load s` is separate: the model is unloaded
first, then a tiny request times the cold load - that is the number you feel
when switching models.

- `decode t/s` is generation throughput (`eval_count / eval_duration`).
- `prefill t/s` is prompt processing (`prompt_eval_count / prompt_eval_duration`).
  The `short` case is only ~10 tokens, so its rate is dominated by overhead;
  compare `medium`/`long` for real prefill throughput.
- `ttft ms` is wall-clock to the first streamed token.
- `vram MiB` / `gpu %` come from `/api/ps` after the run: how much of the model
  is resident in VRAM. `gpu % < 100` means part of it spilled to CPU.

Speed is sensitive to `--num-ctx`, `--num-predict`, thinking mode, other GPU
work, and thermals. Keep those flags identical when comparing. Use
`--no-think` for thinking models when you care about raw throughput; the
default leaves the model's own thinking behaviour in place (which is a real
usage cost, but it makes decode numbers look worse).

**Quality** is a weighted score over the selected suite. Each grader is
deterministic; no LLM judge is involved. Most tasks are pass/fail, but a task
may return partial credit (the webapp task does). A task that hits its token
budget (`done_reason == length`) is always failed and counted in the `trunc`
column. The categories are:

| category | what it checks |
| --- | --- |
| `json` | strict object shape, types, enum-ish values |
| `instruction` | word caps, line counts, exact output shape |
| `code` | generated Python is executed against unit tests, SQL against SQLite |
| `reasoning` | arithmetic, entailment, a small constraint puzzle |
| `extraction` | fields out of messy invoice text |
| `summarize` | hard word cap plus required facts |
| `webapp` | a single-file Snake game: structure, JS syntax, optional browser run |

`--suite quick` runs 8 tasks (~1 min/model); `--suite full` runs all 15.
`--self-test` asserts every grader accepts its reference answer and rejects
empty/garbage output, so a broken grader can't quietly pass everything.

**Score** is `w_quality * quality + w_speed * (decode / fastest decode in this
run)`. Defaults 0.6 / 0.4; override with `--w-quality` / `--w-speed`. Because
speed is normalised within the run, always compare models that were benched
together. `pareto` marks models that nothing else beats on both quality and
speed - the useful frontier when you don't want to pick weights.

## Task filters

`--category` and `--task` select from the full suite regardless of `--suite`,
which is useful for a focused comparison:

```bash
./bench.py -m qwen2.5-coder:3b -m qwen2.5-coder:7b --quality-only --category webapp
./bench.py -m qwen3:4b --quality-only --task code_fizzbuzz --task code_slugify
```

## Webapp tasks and optional browser checks

The `webapp_snake` task asks for a complete single-file Snake game. Its default
grader is static: it extracts the HTML, checks the structure (canvas, input
handling, food, score, game over, restart, loop, collision), rejects external
assets, and runs `node --check` on the inline script when `node` is available.
It returns partial credit as a fraction of checks met, so a nearly-correct game
ranks above one that does not run.

`--browser` adds a Playwright smoke test: load the page in headless Chromium,
confirm the canvas draws and animates, drive the snake into a wall, and attempt
a restart. Runtime behaviour carries 60% of the webapp score, and a console or
page error zeroes that component entirely: a game that throws is not a
partially working game. Screenshots land beside the extracted HTML.

The core tool remains stdlib-only; the browser path is optional:

```bash
cd tools/model-bench
npm install                    # installs Playwright (node_modules is ignored)
npx playwright install chromium  # or point BENCH_CHROME at an existing Chromium
./bench.py -m qwen2.5-coder:7b --quality-only --category webapp --browser
```

If Chromium is already on disk, set `BENCH_CHROME=/path/to/chrome`. `--browser`
fails fast at startup if node/Playwright/Chromium cannot launch, rather than
silently skipping runtime checks.

## Adding models

Edit `models.txt` (one name per line, `#` comments) and use it via
`./bench.py --suggest`; it is only a suggestion list, not auto-installed.

At Q4 on 8 GB VRAM: 3-4B models fit comfortably, 7-8B models fit tightly
(~5 GB, little context headroom), 12B+ spill to CPU and are only useful as a
quality ceiling. Use `vram MiB` and `gpu %` in the report to see what actually
happened on this machine.

## Adding quality tasks

Add a dict to `TASKS` in `tasks.py`:

```python
{
    "id": "my_task",
    "category": "instruction",
    "quick": True,                  # included in --suite quick
    "weight": 1.0,                  # contribution to the quality score
    "num_predict": 200,
    "num_ctx": 4096,                # optional; must fit prompt + output
    "prompt": "...",
    "format": "json",               # optional Ollama format hint
    "grader": my_grader,            # text -> (passed, detail) or (passed, detail, score)
    "reference": "...",             # must pass my_grader in --self-test
    "note": "shown in the report",
}
```

Then run `./bench.py --self-test`. Keep graders tolerant of ` thinking` blocks
and markdown fences (use the helpers in `tasks.py`), but strict about the thing
the task is actually testing. A grader may return a third element, a score in
`[0, 1]`, to earn partial credit. For a runnable webapp, add
`"kind": "webapp"` and `"browser_checks": ["canvas", "animating",
"no_errors", "game_over", "restart"]`.

## Caveats

- **Code execution**: code tasks run model-generated Python in a temporary
  directory with a 10 s timeout and `python3 -I`. It is not a sandbox. Only run
  models you trust, ideally on a machine where that is acceptable.
- **Browser checks** are opt-in and require node + Playwright; without them the
  webapp task is static-only, so a page that parses can still score well while
  failing at runtime. Use `--browser` when that distinction matters.
- **Relative scoring**: `score` is meaningful within one run. Re-run the
  comparison set together before drawing conclusions.
- **WSL2**: GPU clocks and thermal behaviour vary; treat 2-5% differences as
  noise and repeat before chasing them.
- Cloud models (`:cloud`) technically run but their "speed" is network-bound,
  not this computer's; don't mix them into a local comparison.

## Planned improvements

Known gaps, roughly in priority order. None are implemented yet; use the tool as
a per-machine triage harness first.

- **Reproducibility metadata** — record model digest, Ollama environment, and
  GPU driver/CUDA version in `results.json`; a moved tag currently invalidates
  comparisons silently.
- **Uncertainty** — confidence intervals or min–max for speed, and repeated
  samples for quality tasks instead of N=1.
- **Harder coverage** — repo-level/multi-file coding, multi-turn conversations,
  tool calling, and long-context retrieval.
- **Adversarial self-test** — fenced, prose-wrapped, truncated, and near-miss
  outputs, not just a reference answer and garbage.
- **Run comparison** — a `--compare results.json` mode and a baseline view.
- **Degeneration metric** — detect repetition loops, which currently surface
  only as truncation.
- **Composite semantics** — make single-model scores meaningful, or require at
  least two models for a score.
- **Execution isolation** — containerise generated code and browser checks, and
  drop the headless browser's `--no-sandbox`.
