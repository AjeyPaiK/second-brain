#!/usr/bin/env bash
# Install Second Brain user-level systemd services.
# Idempotent — safe to re-run after editing .service files.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/.config/systemd/user"
mkdir -p "$DEST"

cp -f "$HERE/second-brain-api.service" "$DEST/"

# Disable and remove the old watcher service if it exists
systemctl --user disable --now second-brain-watcher.service 2>/dev/null || true
rm -f "$DEST/second-brain-watcher.service"

systemctl --user daemon-reload
systemctl --user enable --now second-brain-api.service

echo
echo "Installed. Status:"
systemctl --user --no-pager status second-brain-api.service | head -6 || true

# Create adapters directory
mkdir -p "$HOME/Projects/second-brain/adapters"

if ! loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes'; then
  echo
  echo ">> User-level services need 'linger' to auto-start on boot."
  echo ">> Run this once (requires sudo):"
  echo "     sudo loginctl enable-linger $USER"
fi

echo
echo "Logs:"
echo "  journalctl --user -u second-brain-api -f"
echo "  tail -f $HOME/Projects/second-brain/logs/api.log"
