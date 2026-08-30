# Muscle Memory — Apple Watch wrist sensor

A tiny watch-only app that streams real wrist IMU (accel + gyro, ~60 Hz) to the
Muscle Memory server over the exact same websocket protocol as `web/phone.html`
(`/ws/phone`) — so the server needs zero changes, the Live Motion tab lights up
the same way, and the watch face shows the top matching task on your wrist.

## Install on your watch (one-time, ~5 min)

1. `open watch/MuscleMemoryWatch.xcodeproj`
2. Select the `MuscleMemoryWatch` target → Signing & Capabilities →
   set **Team** to your personal Apple ID (add it in Xcode → Settings →
   Accounts if it's not there). Free account is fine.
3. Plug in your iPhone (watch paired to it), pick your Apple Watch as the run
   destination, hit **Run**. First install needs you to trust the developer
   cert on the watch: Watch → Settings → General → Device Management.

## Use

1. Start the server on the laptop; note the laptop's LAN IP
   (`ipconfig getifaddr en0`).
2. Watch app → type `that-ip:7860` → **Stream wrist motion**.
   Allow motion + local-network permissions on first run.
3. Mime a task — whisk, hammer, wipe. The big screen's Live Motion tab and the
   watch itself show the matches. Keep your wrist raised while streaming
   (the app streams in the foreground).

Phone hotspot works great: put laptop + watch's iPhone on the hotspot and the
watch follows the phone's network.

## Protocol

Sends `{"samples": [[t, ax, ay, az, gx, gy, gz], ...]}` every 200 ms:
`t` = seconds (monotonic, CMDeviceMotion timestamp), accel = m/s² **including
gravity** (userAcceleration + gravity, ×9.80665), gyro = rad/s — matching
what `phone.html` sends from a browser's devicemotion events.
