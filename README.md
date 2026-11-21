# ai-things

A catalogue of AI coding assets — tools, Claude Code skills, slash commands,
agents, prompts, and MCP patterns — grouped by category and asset name so it is
easy to browse and reuse.

**Website:** <https://yakov.khalinsky.com/ai-things/> — one page per asset.

## Install (one line)

```bash
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh | sh
```

The installer runs in this order:

1. **Tools** — `ol` and `model-bench` into `~/bin`
2. **Skills** — `code-smells` and `comfyui-prompt` into `~/.claude/skills` and
   `~/.claude/commands`
3. **Checkout** — the repo under `~/.local/share/ai-things`

Re-run the same command to update. Restart Claude Code after installing the
skills. Options are environment variables:

| Variable | Effect |
| --- | --- |
| `AI_THINGS_REF` | Branch, tag, or commit to install (default `main`) |
| `AI_THINGS_HOME` | Checkout location (default `~/.local/share/ai-things`) |
| `AI_THINGS_BIN` | Where `ol` and `model-bench` go (default `~/bin`) |
| `AI_THINGS_INSTALL_TOOLS=0` | Skip `ol` and `model-bench` |
| `AI_THINGS_INSTALL_SKILLS=0` | Skip the `~/.claude` skills and commands |
| `AI_THINGS_SOURCE` | Install from a local checkout instead of downloading |

```bash
# skills only, pinned to a commit
curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh \
  | AI_THINGS_REF=<commit> AI_THINGS_INSTALL_TOOLS=0 sh
```

From a checkout, `./scripts/install.sh` installs the skills into the current
project's `.claude/` instead of `~/.claude/`.

## Structure

Each top-level folder is a **category**. Inside, each subfolder is a
self-contained **asset** that owns all of its primitives.

```
ai-things/
├── install.sh                # one-line installer: tools, then skills
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
