#!/usr/bin/env bash
# Install research-code-skills into Claude Code and/or Codex.
# Usage:
#   ./install.sh                 # install into both clients with symlinks
#   ./install.sh claude
#   ./install.sh codex
#   ./install.sh both --copy     # copy instead of symlink
#   ./install.sh both --dry-run  # show planned targets only
set -euo pipefail

SKILL_NAME="research-code-skills"
CLIENT="both"
MODE="link"
DRY_RUN=0

usage() {
  cat >&2 <<'USAGE'
Usage: ./install.sh [both|claude|codex] [--link|--copy] [--dry-run]

Environment overrides:
  CLAUDE_SKILLS_DIR   Install target for Claude Code skills
  CLAUDE_HOME         Base directory for Claude Code, default: ~/.claude
  CODEX_SKILLS_DIR    Single install target for Codex skills
  CODEX_HOME          Base directory for Codex, default: ~/.codex

Defaults:
  Claude Code: ~/.claude/skills
  Codex:       ~/.codex/skills and ~/.agents/skills
USAGE
}

resolve_script_dir() {
  local source="${BASH_SOURCE[0]}"
  local dir
  while [ -L "$source" ]; do
    dir="$(cd -P "$(dirname "$source")" && pwd)"
    source="$(readlink "$source")"
    case "$source" in
      /*) ;;
      *) source="$dir/$source" ;;
    esac
  done
  cd -P "$(dirname "$source")" && pwd
}

case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "install.sh is for macOS/Linux shells. Use WSL or create the skill folder manually on Windows." >&2
    exit 2
    ;;
esac

while [ "$#" -gt 0 ]; do
  case "$1" in
    both|claude|codex)
      CLIENT="$1"
      ;;
    --link)
      MODE="link"
      ;;
    --copy)
      MODE="copy"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

SKILL_DIR="$(resolve_script_dir)"

if [ ! -f "$SKILL_DIR/SKILL.md" ]; then
  echo "SKILL.md not found in $SKILL_DIR" >&2
  exit 1
fi

if ! grep -q "^name: ${SKILL_NAME}$" "$SKILL_DIR/SKILL.md"; then
  echo "Expected SKILL.md frontmatter name: $SKILL_NAME" >&2
  exit 1
fi

install_one() {
  local skills_dir="$1"
  local link="$skills_dir/$SKILL_NAME"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  would install $SKILL_NAME -> $link ($MODE)"
    return 0
  fi

  mkdir -p "$skills_dir"

  if [ "$MODE" = "link" ]; then
    if [ -L "$link" ]; then
      rm "$link"
    elif [ -e "$link" ]; then
      echo "Refusing to overwrite non-symlink path: $link" >&2
      echo "Remove or rename it manually, then rerun this installer." >&2
      return 1
    fi
    ln -s "$SKILL_DIR" "$link"
  else
    if [ -e "$link" ]; then
      echo "Refusing to overwrite existing path: $link" >&2
      echo "Remove or rename it manually, then rerun this installer." >&2
      return 1
    fi
    mkdir -p "$link"
    cp -R "$SKILL_DIR"/. "$link"/
  fi

  if [ ! -f "$link/SKILL.md" ]; then
    echo "Install verification failed: $link/SKILL.md not found" >&2
    return 1
  fi

  echo "  installed $link"
}

install_claude() {
  local skills_dir="${CLAUDE_SKILLS_DIR:-${CLAUDE_HOME:-$HOME/.claude}/skills}"
  echo "Claude Code:"
  install_one "$skills_dir"
}

install_codex() {
  if [ -n "${CODEX_SKILLS_DIR:-}" ]; then
    echo "Codex:"
    install_one "$CODEX_SKILLS_DIR"
  else
    local codex_home_skills="${CODEX_HOME:-$HOME/.codex}/skills"
    local agents_skills="$HOME/.agents/skills"
    echo "Codex:"
    install_one "$codex_home_skills"
    if [ "$agents_skills" != "$codex_home_skills" ]; then
      install_one "$agents_skills"
    fi
  fi
}

case "$CLIENT" in
  both)
    install_claude
    install_codex
    ;;
  claude)
    install_claude
    ;;
  codex)
    install_codex
    ;;
esac

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete."
else
  echo "Done. Restart Claude Code or Codex to discover the skill."
fi
