#!/usr/bin/env python3
"""Bridge glasses jog commands to the arm.

Run this on whatever machine holds the arm's USB cable. It subscribes to
/ws/jog/control and hands each command to `drive()`, which is the one function
to fill in with the partner-side arrow-key-to-movement code.

    # dry run first -- prints commands, touches no hardware
    python scripts/jog_bridge.py --server http://localhost:7860

    # arm on a different machine to the server: point at the tunnel
    python scripts/jog_bridge.py --server https://<name>.trycloudflare.com

Same machine as the server is the simpler setup and the lower-latency one:
localhost skips the round trip out to Cloudflare and back that the glasses
already pay once.

Commands arrive as {seq, mode, dir, axis}. Drive off `axis` -- "+X", "-Y",
"+Z", "GRIP+" -- not `dir`. `dir` is only which way the operator swiped, and
means different things depending on which mode the glasses are in.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")


# --- fill this in -----------------------------------------------------------
# Map an axis to your arm. Values are a nudge per press, not a velocity: the
# glasses send discrete jogs, so each command should produce one bounded move
# and stop. Sizes are in whatever unit the partner-side movement code takes.
STEP = {
    "+X": ("x", +1), "-X": ("x", -1),
    "+Y": ("y", +1), "-Y": ("y", -1),
    "+Z": ("z", +1), "-Z": ("z", -1),
    "GRIP+": ("grip", +1), "GRIP-": ("grip", -1),
}


def drive(axis: str, step: int, cmd: dict) -> None:
    """Move the arm one step. Replace the body with the real call.

    Deliberately synchronous and blocking: finishing one jog before the next is
    read gives back-pressure for free. Commands that arrive mid-move queue in
    the socket, and since a stale jog is worth nothing, dropping them is safer
    than replaying them late -- see --skip-stale.
    """
    print(f"  drive {axis:>4} {step:+d}   (seq {cmd['seq']}, mode {cmd['mode']})",
          flush=True)
    # from partner_module import jog
    # jog(axis, step)


# ----------------------------------------------------------------------------
async def run(url: str, skip_stale: bool) -> None:
    while True:
        try:
            async with websockets.connect(url, open_timeout=10) as ws:
                print(f"connected: {url}", flush=True)
                while True:
                    # recv() throughout rather than `async for`: iterating the
                    # socket and calling recv() on it compete for the same
                    # reader, and the draining below needs recv().
                    cmd = json.loads(await ws.recv())
                    # Anything already queued behind this one is out of date;
                    # the operator has moved on. Take the newest, drop the rest.
                    if skip_stale:
                        while True:
                            try:
                                cmd = json.loads(
                                    await asyncio.wait_for(ws.recv(), timeout=0.001))
                            except asyncio.TimeoutError:
                                break
                    mapped = STEP.get(cmd.get("axis"))
                    if mapped is None:
                        print(f"  ignored unknown axis {cmd.get('axis')!r}", flush=True)
                        continue
                    axis, step = mapped
                    drive(axis, step, cmd)
        except Exception as exc:
            print(f"disconnected ({type(exc).__name__}: {exc}); retrying in 2s",
                  flush=True)
            await asyncio.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:7860",
                    help="base URL of the Muscle Memory server")
    ap.add_argument("--skip-stale", action="store_true", default=True,
                    help="on a backlog, act on the newest command only (default)")
    ap.add_argument("--no-skip-stale", dest="skip_stale", action="store_false")
    args = ap.parse_args()

    base = args.server.rstrip("/")
    url = ("wss://" + base[len("https://"):]) if base.startswith("https://") \
        else ("ws://" + base[len("http://"):])
    url += "/ws/jog/control"

    try:
        asyncio.run(run(url, args.skip_stale))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
