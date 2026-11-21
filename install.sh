#!/bin/sh
# ai-things installer
#
# Nothing is installed unless you pick it. Usage:
#
#   curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh \
#     | sh -s -- <selection> [<selection>...]
#
# Selections:
#   ol              ollama agent launcher        -> ~/bin/ol
#   model-bench     local model speed/quality    -> ~/bin/model-bench
#   code-smells     Claude Code skill + command
#   comfyui-prompt  Claude Code skill + commands
#   tools           ol and model-bench
#   skills          code-smells and comfyui-prompt
#   all             everything above
#
# Examples:
#   ... | sh -s -- ol
#   ... | sh -s -- model-bench
#   ... | sh -s -- ol model-bench
#   ... | sh -s -- code-smells
#   ... | sh -s -- all
#
# Or set AI_THINGS_INSTALL="ol,model-bench" instead of arguments.
#
# Other options:
#   AI_THINGS_REF=main              git ref / branch / commit to install
#   AI_THINGS_HOME=...              checkout location (default ~/.local/share/ai-things)
#   AI_THINGS_BIN=~/bin             where ol and model-bench go
#   AI_THINGS_SOURCE=/path/to/repo  install from a local checkout (no download)

set -eu

REPO_SLUG="yakovkhalinsky/ai-things"
REF="${AI_THINGS_REF:-main}"
HOME_DIR="${AI_THINGS_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/ai-things}"
BIN_DIR="${AI_THINGS_BIN:-${HOME}/bin}"
SOURCE="${AI_THINGS_SOURCE:-}"

say() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
ai-things installer

Nothing is installed unless you pick it:

  ol              ollama agent launcher        -> ~/bin/ol
  model-bench     local model speed/quality    -> ~/bin/model-bench
  code-smells     Claude Code skill + command
  comfyui-prompt  Claude Code skill + commands
  tools           ol and model-bench
  skills          code-smells and comfyui-prompt
  all             everything above

Usage:
  curl -fsSL https://raw.githubusercontent.com/yakovkhalinsky/ai-things/main/install.sh \
    | sh -s -- <selection> [<selection>...]

Examples:
  ... | sh -s -- ol
  ... | sh -s -- model-bench
  ... | sh -s -- ol model-bench
  ... | sh -s -- code-smells
  ... | sh -s -- all

Environment:
  AI_THINGS_INSTALL="ol,model-bench"   selections without arguments
  AI_THINGS_REF=main                   branch, tag, or commit
  AI_THINGS_HOME=...                   checkout location
  AI_THINGS_BIN=~/bin                  where ol and model-bench go
  AI_THINGS_SOURCE=/path/to/repo       install from a local checkout
EOF
}

SELECTED=""
add_selection() {
    case " ${SELECTED} " in
        *" $1 "*) ;;
        *) SELECTED="${SELECTED} $1" ;;
    esac
}

select_item() {
    case "$1" in
        ol|model-bench|code-smells|comfyui-prompt)
            add_selection "$1"
            ;;
        tools)
            add_selection ol
            add_selection model-bench
            ;;
        skills)
            add_selection code-smells
            add_selection comfyui-prompt
            ;;
        all)
            add_selection ol
            add_selection model-bench
            add_selection code-smells
            add_selection comfyui-prompt
            ;;
        *)
            die "unknown selection: $1 (try --help)"
            ;;
    esac
}

wants() {
    case " ${SELECTED} " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}

for arg in "$@"; do
    case "$arg" in
        -h|--help|help)
            usage
            exit 0
            ;;
    esac
    case "$arg" in
        --*) arg=${arg#--} ;;
    esac
    select_item "$arg"
done

if [ -z "${SELECTED}" ] && [ -n "${AI_THINGS_INSTALL:-}" ]; then
    for item in $(printf '%s' "${AI_THINGS_INSTALL}" | tr ',' ' '); do
        select_item "$item"
    done
fi

if [ -z "${SELECTED}" ]; then
    usage >&2
    exit 1
fi

TMP=""
cleanup() {
    if [ -n "${TMP}" ]; then
        rm -rf "${TMP}"
    fi
}
trap cleanup EXIT INT TERM

if [ -z "${SOURCE}" ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required"
    command -v tar >/dev/null 2>&1 || die "tar is required"
fi

if [ -n "${SOURCE}" ]; then
    [ -d "${SOURCE}" ] || die "AI_THINGS_SOURCE is not a directory: ${SOURCE}"
    SRC="${SOURCE}"
    say "Using local source: ${SRC}"
else
    TMP="$(mktemp -d 2>/dev/null || mktemp -d -t ai-things)"
    URL="https://codeload.github.com/${REPO_SLUG}/tar.gz/${REF}"
    say "Downloading ${URL}"
    curl -fsSL "${URL}" -o "${TMP}/repo.tar.gz" || die "download failed (ref: ${REF})"
    tar -xzf "${TMP}/repo.tar.gz" -C "${TMP}" || die "could not extract archive"
    SRC="$(find "${TMP}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -n "${SRC}" ] || die "archive did not contain a directory"
fi

# Stage the checkout, then swap it in. A failed download or copy must not leave
# a half-written install behind.
STAGE="${HOME_DIR}.new.$$"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp -R "${SRC}/." "${STAGE}/"
rm -rf "${STAGE}/.git" "${STAGE}/.claude" \
       "${STAGE}/tools/model-bench/results" "${STAGE}/tools/model-bench/node_modules"
find "${STAGE}" -type d -name __pycache__ -prune -exec rm -rf {} \; 2>/dev/null || true
rm -rf "${HOME_DIR}"
mkdir -p "$(dirname "${HOME_DIR}")"
mv "${STAGE}" "${HOME_DIR}"
say "Checkout: ${HOME_DIR}"

# --- tools -----------------------------------------------------------------

install_model_bench() {
    target="${BIN_DIR}/model-bench"
    tmp="${target}.new.$$"
    cat >"${tmp}" <<EOF
#!/bin/sh
# ai-things model-bench wrapper
exec python3 "${HOME_DIR}/tools/model-bench/bench.py" "\$@"
EOF
    chmod +x "${tmp}"
    if [ -e "${target}" ] || [ -L "${target}" ]; then
        if ! grep -qF 'ai-things model-bench wrapper' "${target}" 2>/dev/null; then
            mv "${target}" "${target}.bak.$(date +%s)"
            say "Backed up existing ${target}"
        fi
    fi
    mv "${tmp}" "${target}"
    say "Installed ${target}"
}

OL_OK=0
MB_OK=0
if wants ol || wants model-bench; then
    mkdir -p "${BIN_DIR}"

    if wants ol; then
        if command -v bash >/dev/null 2>&1; then
            if OL_TARGET="${BIN_DIR}/ol" bash "${HOME_DIR}/tools/ol/install.sh"; then
                OL_OK=1
            else
                warn "ol install failed; continuing"
            fi
        else
            warn "bash not found; skipping ol"
        fi
    fi

    if wants model-bench; then
        if command -v python3 >/dev/null 2>&1; then
            if install_model_bench; then
                MB_OK=1
            else
                warn "model-bench install failed; continuing"
            fi
        else
            warn "python3 not found; skipping model-bench"
        fi
    fi
fi

# ol's installer owns the ~/bin PATH line when it runs. Only add one if it is
# still missing, and only for the default ~/bin (a custom directory is the
# user's business).
ensure_path() {
    [ "${BIN_DIR}" = "${HOME}/bin" ] || {
        say "Note: ${BIN_DIR} is not ~/bin; make sure it is on PATH."
        return 0
    }
    rc=""
    case "$(basename "${SHELL:-}")" in
        zsh) rc="${HOME}/.zshrc" ;;
        bash) rc="${HOME}/.bashrc" ;;
        *)
            if [ -f "${HOME}/.bashrc" ]; then
                rc="${HOME}/.bashrc"
            elif [ -f "${HOME}/.zshrc" ]; then
                rc="${HOME}/.zshrc"
            elif [ -f "${HOME}/.profile" ]; then
                rc="${HOME}/.profile"
            fi
            ;;
    esac
    if [ -z "${rc}" ]; then
        say "Add this to your shell rc: export PATH=\"\$HOME/bin:\$PATH\""
        return 0
    fi
    if grep -qF '${HOME}/bin' "${rc}" 2>/dev/null || grep -qF "${HOME}/bin" "${rc}" 2>/dev/null; then
        return 0
    fi
    printf '\n# Added by ai-things install.sh\nexport PATH="${HOME}/bin:${PATH}"\n' >>"${rc}"
    say "Added ~/bin to ${rc}"
    PATH_RC="${rc}"
}
PATH_RC=""
if [ "${OL_OK}" = "1" ] || [ "${MB_OK}" = "1" ]; then
    ensure_path
fi

# --- skills and commands ---------------------------------------------------

backup_if_real() {
    target="$1"
    if [ -L "${target}" ]; then
        rm -f "${target}"
    elif [ -e "${target}" ]; then
        mv "${target}" "${target}.bak.$(date +%s)"
        say "Backed up existing ${target}"
    fi
}

link_skill() {
    name="$1"
    src="$2"
    target="${HOME}/.claude/skills/${name}"
    mkdir -p "${HOME}/.claude/skills"
    backup_if_real "${target}"
    if ln -s "${src}" "${target}" 2>/dev/null; then
        say "Linked skill ${name}"
    else
        mkdir -p "${target}"
        cp -R "${src}/." "${target}/"
        say "Copied skill ${name} (symlinks unavailable)"
    fi
}

link_command() {
    name="$1"
    src="$2"
    target="${HOME}/.claude/commands/${name}"
    mkdir -p "${HOME}/.claude/commands"
    backup_if_real "${target}"
    if ln -s "${src}" "${target}" 2>/dev/null; then
        say "Linked command /${name%.md}"
    else
        cp "${src}" "${target}"
        say "Copied command /${name%.md} (symlinks unavailable)"
    fi
}

SKILLS_OK=0
if wants code-smells; then
    link_skill code-smells "${HOME_DIR}/code-quality/code-smells"
    link_command code-smells.md "${HOME_DIR}/code-quality/code-smells/code-smells.md"
    SKILLS_OK=1
fi
if wants comfyui-prompt; then
    link_skill comfyui-prompt "${HOME_DIR}/image-generation/comfyui-prompt"
    link_command comfyui-prompt.md "${HOME_DIR}/image-generation/comfyui-prompt/comfyui-prompt.md"
    link_command comfyui-serve-output.md "${HOME_DIR}/image-generation/comfyui-prompt/command-serve-output.md"
    SKILLS_OK=1
fi

say ""
say "ai-things installed."
if [ "${OL_OK}" = "1" ]; then
    say "  ol:          ${BIN_DIR}/ol"
fi
if [ "${MB_OK}" = "1" ]; then
    say "  model-bench: ${BIN_DIR}/model-bench"
fi
if wants code-smells; then
    say "  skill:       code-smells, /code-smells"
fi
if wants comfyui-prompt; then
    say "  skill:       comfyui-prompt, /comfyui-prompt, /comfyui-serve-output"
fi
if [ "${SKILLS_OK}" = "1" ]; then
    say "  restart Claude Code to pick up the skills."
fi
say "Update by re-running the same selection."
if [ -n "${PATH_RC}" ]; then
    say "Reload your shell: source ${PATH_RC}"
fi
