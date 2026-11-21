# ai-things

A catalogue of AI coding assets — tools, Claude Code skills, slash commands,
agents, prompts, and MCP patterns — grouped by category and asset name so it is
easy to browse and reuse.

**Website:** <https://yakov.khalinsky.com/ai-things/> — one page per asset.

## Install (one line, pick what you want)

Nothing is installed unless you select it. Pass selections after `sh -s --`:

```bash
# one tool
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- ol
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- model-bench

# one skill
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- code-smells
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- comfyui-prompt

# shortcuts
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- tools
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- skills
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh -s -- all
```

Each run also installs the checkout under `~/.local/share/ai-things`. Tools go
to `~/bin`; skills and commands go to `~/.claude`. Re-run the same selection to
update, and restart Claude Code after installing skills.

| Selection | Installs |
| --- | --- |
| `ol` | `~/bin/ol` |
| `model-bench` | `~/bin/model-bench` |
| `code-smells` | skill + `/code-smells` command |
| `comfyui-prompt` | skill + `/comfyui-prompt`, `/comfyui-serve-output` |
| `tools` | `ol` + `model-bench` |
| `skills` | `code-smells` + `comfyui-prompt` |
| `all` | everything above |

Options are environment variables:

| Variable | Effect |
| --- | --- |
| `AI_THINGS_INSTALL` | Selections without arguments, e.g. `ol,model-bench` |
| `AI_THINGS_REF` | Branch, tag, or commit to install (default `main`) |
| `AI_THINGS_HOME` | Checkout location (default `~/.local/share/ai-things`) |
| `AI_THINGS_BIN` | Where `ol` and `model-bench` go (default `~/bin`) |
| `AI_THINGS_SOURCE` | Install from a local checkout instead of downloading |

```bash
# environment form, pinned to a commit (pin the script URL too for full reproducibility)
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/<commit>/install.sh \
  | AI_THINGS_REF=<commit> AI_THINGS_INSTALL=model-bench sh
```

From a checkout, `./scripts/install.sh` installs the skills into the current
project's `.claude/` instead of `~/.claude/`.

## Structure

Each top-level folder is a **category**. Inside, each subfolder is a
self-contained **asset** that owns all of its primitives.

```
ai-things/
├── install.sh                # one-line installer; select tools/skills
├── tools/
│   ├── ol/                   # Ollama agent launcher
│   │   ├── ol
│   │   ├── ol-completion.bash
│   │   ├── ol-completion.zsh
│   │   ├── test-completion.sh
│   │   ├── install.sh
│   │   └── README.md
│   └── model-bench/          # local model speed vs quality harness
│       ├── bench.py
│       ├── tasks.py
│       ├── browser_smoke.mjs # optional Playwright runtime checks
│       ├── models.txt
│       └── README.md
├── code-quality/
│   └── code-smells/
│       ├── README.md
│       ├── SKILL.md
│       └── code-smells.md
├── image-generation/
│   └── comfyui-prompt/
│       ├── README.md
│       ├── SKILL.md
│       ├── comfyui-prompt.md
│       ├── command-serve-output.md
│       └── workflows/
├── agents/  prompts/  mcp/   # placeholders
├── scripts/install.sh        # project-local skills installer
└── .claude/                  # project-local symlinks into assets
```

The root `.claude/` directory contains symlinks to the active `SKILL.md`,
command files, etc. inside each asset folder. On systems without symlink
support, run `./scripts/install.sh` and restart Claude Code.

## What's here

| Category | Asset | Description |
|----------|-------|-------------|
| tools | [`ol`](tools/ol/) | Ollama agent launcher that syncs itself from the remote repo, with bash/zsh tab completion |
| tools | [`model-bench`](tools/model-bench/) | Speed vs quality harness for trialling local Ollama models, with an optional headless-browser check |
| code-quality | [`code-smells`](code-quality/code-smells/) | Catalogue of Martin Fowler's 24 code smells, with symptoms and treatments |
| image-generation | [`comfyui-prompt`](image-generation/comfyui-prompt/) | Build and run ComfyUI image-generation prompts from natural language |

## Adding a new item

1. Pick or create a category folder, e.g. `image-generation/` or `tools/`.
2. Create an asset subfolder, e.g. `tools/my-thing/`.
3. Add the required Claude Code file(s) (`SKILL.md`, `command.md`, `AGENT.md`,
   etc.) and a human-readable `README.md`.
4. If it is a skill or command, add it to `scripts/install.sh` and the root
   `install.sh` skill list, plus a symlink in `.claude/`.
5. Add a page under `docs/` and link it from `docs/index.html`.
6. Update the table above.

## License

MIT — use, fork, and improve freely.
