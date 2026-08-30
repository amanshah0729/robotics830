#!/usr/bin/env bash
# Expose the local Muscle Memory server over HTTPS.
#
# Meta Ray-Ban Display refuses plain HTTP: the glasses runtime requires a
# publicly reachable https:// URL for every Web App it loads, and the page is
# fetched through the paired phone rather than off the local network — so a
# LAN address like http://192.168.1.x:7860 is unreachable twice over. iOS
# Safari also gates DeviceMotionEvent behind HTTPS, so the phone client at
# /phone needs this too.
#
#   ./scripts/tunnel.sh            # tunnel the default port
#   ./scripts/tunnel.sh 8000       # tunnel some other port
#
# Take the printed https://<something>.trycloudflare.com URL and add it in the
# Meta AI app: App Settings > App Connections > Web Apps > Add a Web App,
# pointing at <url>/glasses. Developer Mode must already be on (Settings >
# App Info > tap App version five times).
set -euo pipefail

PORT="${1:-7860}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install it with:" >&2
  echo "  brew install cloudflared" >&2
  exit 1
fi

if ! curl -sf -o /dev/null --max-time 2 "http://localhost:${PORT}/"; then
  echo "Nothing is answering on http://localhost:${PORT}" >&2
  echo "Start the server first, e.g.:" >&2
  echo "  python -m musclememory.server --source fixture --port ${PORT}" >&2
  exit 1
fi

cat <<BANNER

  Tunnelling localhost:${PORT} over HTTPS.
  Watch for the trycloudflare.com URL below, then load these:

    <url>/glasses    Meta Ray-Ban Display  (add via Meta AI app)
    <url>/phone      phone / handheld motion
    <url>/           atlas, for the big screen

  The URL changes on every restart, so re-add the Web App if you restart this.

BANNER

exec cloudflared tunnel --url "http://localhost:${PORT}"
