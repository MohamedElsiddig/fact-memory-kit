#!/usr/bin/env bash
# Install the factkit companion plugin for Hermes, alongside holographic memory.
#
#   bash install.sh           install/refresh the plugin
#   bash install.sh --check   verify the installed copy matches this checkout
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: install.sh [--check]

  (no args)  install or refresh $HOME/.hermes/plugins/factkit
  --check    compare the installed plugin files with this checkout; prints
             OK, or one DRIFT <file> line per mismatch, and exits 1 on drift.
             Never writes anything.
USAGE
}

check=0
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "install.sh: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
plugins="${HERMES_PLUGINS:-$HOME/.hermes/plugins}"
dest="$plugins/factkit"

manifest="$here/manifest.txt"
if [ ! -f "$manifest" ]; then
  echo "install.sh: manifest not found: $manifest" >&2
  exit 2
fi

mapfile -t entries < <(
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -n "$line" ] && printf '%s\n' "$line"
  done < "$manifest"
)

# installed path (relative to $dest) -> repo source
source_for() {
  case "$1" in
    __init__.py) printf '%s\n' "$here/plugin_init.py" ;;
    plugin.yaml) printf '%s\n' "$here/plugin.yaml" ;;
    manifest.txt) printf '%s\n' "$here/manifest.txt" ;;
    factkit/*.py) printf '%s\n' "$repo/$1" ;;
    *) printf '%s\n' "" ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# files owned by the plugin's own update channel, not by this manifest:
# they may legitimately appear next to an installed copy and are not drift
is_dist_owned() {
  case "$1" in
    update.yaml|README.md|LICENSE|config.yaml.example|config.example.yaml|config.yaml|facts.db|facts.db-shm|facts.db-wal|*.pyc|__pycache__)
      return 0 ;;
  esac
  return 1
}

if [ "$check" -eq 1 ]; then
  drift=0
  declare -A in_manifest=()
  for rel in "${entries[@]}"; do
    in_manifest["$rel"]=1
    src="$(source_for "$rel")"
    dst="$dest/$rel"
    if [ -z "$src" ] || [ ! -f "$src" ]; then
      echo "DRIFT $rel"
      echo "  missing repo source: ${src:-<unmapped>}" >&2
      drift=1
      continue
    fi
    if [ ! -f "$dst" ]; then
      echo "DRIFT $rel"
      echo "  missing installed copy: $dst" >&2
      drift=1
      continue
    fi
    if [ "$(sha256 "$src")" != "$(sha256 "$dst")" ]; then
      echo "DRIFT $rel"
      drift=1
    fi
  done

  # generated entry point; not in the manifest but plugin-status checks it too
  if [ ! -f "$dest/factkit/__init__.py" ]; then
    echo "DRIFT factkit/__init__.py"
    echo "  missing generated copy: $dest/factkit/__init__.py" >&2
    drift=1
  fi

  # extra files in the plugin root: seeded config, dist-owned files and bytecode are expected
  if [ -d "$dest" ]; then
    for f in "$dest"/*; do
      [ -f "$f" ] || continue
      base="$(basename "$f")"
      if is_dist_owned "$base"; then
        if [ "${FACTKIT_CHECK_VERBOSE:-0}" = "1" ]; then
          echo "IGNORED $base"
        fi
        continue
      fi
      if [ -z "${in_manifest[$base]+x}" ]; then
        echo "DRIFT $base"
        drift=1
      fi
    done
  fi

  # extra files in the vendored package: generated entry point and dist-owned files are expected
  if [ -d "$dest/factkit" ]; then
    for f in "$dest/factkit"/*; do
      [ -f "$f" ] || continue
      base="$(basename "$f")"
      if [ "$base" = "__init__.py" ] || is_dist_owned "$base"; then
        if [ "${FACTKIT_CHECK_VERBOSE:-0}" = "1" ]; then
          echo "IGNORED factkit/$base"
        fi
        continue
      fi
      rel="factkit/$base"
      if [ -z "${in_manifest[$rel]+x}" ]; then
        echo "DRIFT $rel"
        drift=1
      fi
    done
  fi

  if [ "$drift" -eq 0 ]; then
    echo "OK"
    exit 0
  fi
  echo "drift detected in $dest — re-run install.sh to resync" >&2
  exit 1
fi

mkdir -p "$dest/factkit"

# manifest-driven copy: plugin entry point, manifest, and vendored modules
for rel in "${entries[@]}"; do
  src="$(source_for "$rel")"
  if [ -z "$src" ]; then
    echo "install.sh: no source mapping for manifest entry: $rel" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$dest/$rel")"
  cp "$src" "$dest/$rel"
done
cat > "$dest/factkit/__init__.py" <<'PY'
"""Minimal factkit subset bundled with the Hermes plugin (holo + rerank + helpers)."""
__all__ = ["holo", "rerank"]
__version__ = "0.1.0"
PY

if [ ! -f "$dest/config.yaml" ]; then
  cp "$here/config.example.yaml" "$dest/config.yaml"
  chmod 600 "$dest/config.yaml"
  echo "wrote $dest/config.yaml — add embed_api_key if your endpoint needs one"
fi

echo "installed to $dest"
echo "next: hermes plugins enable factkit"
