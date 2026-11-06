#!/usr/bin/env bash
# Install the factkit companion plugin for Hermes, alongside holographic memory.
#
#   bash install.sh           install/refresh the plugin
#   bash install.sh --check   verify the installed copy matches this checkout
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: install.sh [--check]

  (no args)  install or refresh ~/.hermes/plugins/factkit
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

# vendored files: installed path -> repo source
labels=(__init__.py plugin.yaml factkit/config.py factkit/embed.py factkit/vectors.py factkit/units.py factkit/embedcache.py factkit/maintenance.py factkit/holo.py)
sources=(
  "$here/plugin_init.py"
  "$here/plugin.yaml"
  "$repo/factkit/config.py"
  "$repo/factkit/embed.py"
  "$repo/factkit/vectors.py"
  "$repo/factkit/units.py"
  "$repo/factkit/embedcache.py"
  "$repo/factkit/maintenance.py"
  "$repo/factkit/holo.py"
)
targets=(
  "$dest/__init__.py"
  "$dest/plugin.yaml"
  "$dest/factkit/config.py"
  "$dest/factkit/embed.py"
  "$dest/factkit/vectors.py"
  "$dest/factkit/units.py"
  "$dest/factkit/embedcache.py"
  "$dest/factkit/maintenance.py"
  "$dest/factkit/holo.py"
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [ "$check" -eq 1 ]; then
  drift=0
  for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    src="${sources[$i]}"
    dst="${targets[$i]}"
    if [ ! -f "$src" ]; then
      echo "DRIFT $label"
      echo "  missing repo source: $src" >&2
      drift=1
      continue
    fi
    if [ ! -f "$dst" ]; then
      echo "DRIFT $label"
      echo "  missing installed copy: $dst" >&2
      drift=1
      continue
    fi
    if [ "$(sha256 "$src")" != "$(sha256 "$dst")" ]; then
      echo "DRIFT $label"
      drift=1
    fi
  done
  if [ "$drift" -eq 0 ]; then
    echo "OK"
    exit 0
  fi
  echo "drift detected in $dest — re-run install.sh to resync" >&2
  exit 1
fi

mkdir -p "$dest/factkit"

# plugin entry point + manifest
cp "$here/plugin_init.py" "$dest/__init__.py"
cp "$here/plugin.yaml" "$dest/plugin.yaml"

# vendored subset: holo machinery only, no CLI or server
for f in config.py embed.py vectors.py units.py embedcache.py maintenance.py holo.py; do
  cp "$repo/factkit/$f" "$dest/factkit/$f"
done
cat > "$dest/factkit/__init__.py" <<'PY'
"""Minimal factkit subset bundled with the Hermes plugin (holo + helpers)."""
__all__ = ["holo"]
__version__ = "0.1.0"
PY

if [ ! -f "$dest/config.yaml" ]; then
  cp "$here/config.example.yaml" "$dest/config.yaml"
  chmod 600 "$dest/config.yaml"
  echo "wrote $dest/config.yaml — add embed_api_key if your endpoint needs one"
fi

echo "installed to $dest"
echo "next: hermes plugins enable factkit"
