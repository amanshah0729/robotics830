"""Feed Meta Ray-Ban Display jog commands into a keyboard teleop loop.

Deliberately stdlib-only. This runs inside the lerobot venv on whatever machine
holds the arm's USB cable, and adding a websocket client there would mean a pip
install on the demo machine; long-polling over urllib needs nothing.

A background thread polls the server and, for each command, holds the mapped
key down for a moment in the same set the keyboard listener writes to. The
teleop loop then reads glasses input through exactly the code path it already
uses for the keyboard -- speed scaling, leashing, IK and mode handling all
apply unchanged, and the two input sources compose rather than compete.

Holding matters: the glasses emit one discrete event per swipe, while the
teleop loop integrates *held* keys every tick. A press with no duration would
move the arm for a single frame and look like nothing happened.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class GlassesJog:
    """Injects glasses jog commands into a shared pressed-key set."""

    def __init__(self, server: str, keymap: dict, pressed: set,
                 hold_s: float = 0.25, on_event=None) -> None:
        self.base = server.rstrip("/")
        self.keymap = keymap          # {"+X": <key>, "GRIP-": <key>, ...}
        self.pressed = pressed        # the teleop loop's live key set
        self.hold_s = hold_s
        self.on_event = on_event or (lambda msg: None)
        self._held: dict = {}         # key -> release deadline
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.connected = False
        self.count = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "GlassesJog":
        for target in (self._poll, self._release_expired):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        for key in list(self._held):
            self.pressed.discard(key)
        self._held.clear()

    # -- internals ----------------------------------------------------------
    def _seed_after(self) -> int:
        """Start from the current head of the log, not the beginning of it.

        Polling from zero replays whatever was issued last -- possibly minutes
        ago, from a previous session -- and the arm lurches the moment this
        connects. Commands are only meaningful live, so history is skipped.
        """
        try:
            with urllib.request.urlopen(f"{self.base}/api/jog/log?limit=1",
                                        timeout=10) as r:
                recent = json.loads(r.read() or b"{}").get("recent") or []
            return int(recent[-1].get("seq", 0)) if recent else 0
        except Exception:
            return 0

    def _poll(self) -> None:
        after = self._seed_after()
        while not self._stop.is_set():
            try:
                url = (f"{self.base}/api/jog/next?"
                       + urllib.parse.urlencode({"after": after, "timeout": 20}))
                with urllib.request.urlopen(url, timeout=30) as r:
                    cmd = json.loads(r.read() or b"{}")
                if not self.connected:
                    self.connected = True
                    self.on_event(f"glasses connected: {self.base}")
                if not cmd:
                    continue                     # idle timeout, just poll again
                after = cmd.get("seq", after)
                key = self.keymap.get(cmd.get("axis"))
                if key is None:
                    self.on_event(f"glasses: unmapped axis {cmd.get('axis')!r}")
                    continue
                self.count += 1
                # Extend rather than restart, so repeated swipes in the same
                # direction produce continuous motion instead of stutter.
                self._held[key] = time.monotonic() + self.hold_s
                self.pressed.add(key)
                self.on_event(f"glasses {cmd.get('axis')}  ({cmd.get('mode')})")
            except urllib.error.URLError as exc:
                if self.connected or after == 0:
                    self.on_event(f"glasses link down ({exc.reason}); retrying")
                self.connected = False
                self._stop.wait(2.0)
            except Exception as exc:             # malformed payload, etc.
                self.on_event(f"glasses error: {type(exc).__name__}: {exc}")
                self._stop.wait(1.0)

    def _release_expired(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            for key, deadline in list(self._held.items()):
                if now >= deadline:
                    self._held.pop(key, None)
                    self.pressed.discard(key)
            self._stop.wait(0.01)
