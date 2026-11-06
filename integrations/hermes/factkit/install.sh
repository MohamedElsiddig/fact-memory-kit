#!/usr/bin/env bash
# Install the factkit companion plugin for Hermes, alongside holographic memory.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
plugins="${HERMES_PLUGINS:-$HOME/.hermes/plugins}"
dest="$plugins/factkit"

mkdir -p "$dest/factkit"

# plugin entry point + manifest
cp "$here/plugin_init.py" "$dest/__init__.py"
cp "$here/plugin.yaml" "$dest/plugin.yaml"

# vendored subset: holo machinery only, no CLI or server
for f in config.py embed.py vectors.py holo.py; do
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
