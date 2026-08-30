import CoreMotion
import Foundation

/// Streams wrist IMU to the Muscle Memory server over the same websocket
/// protocol as web/phone.html: {"samples": [[t, ax, ay, az, gx, gy, gz], ...]}
/// with accel in m/s^2 (gravity included) and gyro in rad/s.
final class MotionStreamer: NSObject, ObservableObject {
    @Published var running = false
    @Published var status = "idle"
    @Published var topMatch = ""
    @Published var energy: Double = 0

    private let motion = CMMotionManager()
    private var socket: URLSessionWebSocketTask?
    private var flushTimer: Timer?
    private var buf: [[Double]] = []
    private let lock = NSLock()
    private static let g = 9.80665

    func start(host: String) {
        stop()
        let clean = host.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "http://", with: "")
            .replacingOccurrences(of: "ws://", with: "")
        guard let url = URL(string: "ws://\(clean)/ws/phone") else {
            status = "bad server address"
            return
        }
        running = true
        status = "connecting…"
        connect(url)
        startMotion()
        flushTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.flush()
        }
    }

    func stop() {
        running = false
        motion.stopDeviceMotionUpdates()
        flushTimer?.invalidate(); flushTimer = nil
        socket?.cancel(with: .goingAway, reason: nil); socket = nil
        lock.lock(); buf.removeAll(); lock.unlock()
        status = "idle"
        topMatch = ""
        energy = 0
    }

    // MARK: - motion

    private func startMotion() {
        guard motion.isDeviceMotionAvailable else {
            status = "no motion sensors?!"
            return
        }
        motion.deviceMotionUpdateInterval = 1.0 / 60.0
        motion.startDeviceMotionUpdates(to: .main) { [weak self] dm, _ in
            guard let self, let dm else { return }
            let ax = (dm.userAcceleration.x + dm.gravity.x) * Self.g
            let ay = (dm.userAcceleration.y + dm.gravity.y) * Self.g
            let az = (dm.userAcceleration.z + dm.gravity.z) * Self.g
            let r = dm.rotationRate
            self.lock.lock()
            self.buf.append([dm.timestamp, ax, ay, az, r.x, r.y, r.z])
            self.lock.unlock()
            let user = dm.userAcceleration
            let e = sqrt(user.x * user.x + user.y * user.y + user.z * user.z)
            self.energy = min(1.0, e / 1.5)
        }
    }

    private func flush() {
        lock.lock()
        let samples = buf
        buf.removeAll()
        lock.unlock()
        guard let socket, !samples.isEmpty else { return }
        let rounded = samples.map { $0.map { (v: Double) in (v * 1000).rounded() / 1000 } }
        guard let data = try? JSONSerialization.data(withJSONObject: ["samples": rounded]),
              let text = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(text)) { [weak self] err in
            if err != nil {
                DispatchQueue.main.async { self?.status = "send failed — reconnecting…" }
            }
        }
    }

    // MARK: - websocket

    private func connect(_ url: URL) {
        let task = URLSession.shared.webSocketTask(with: url)
        socket = task
        task.resume()
        DispatchQueue.main.async { self.status = "streaming — move your wrist!" }
        receive(task)
    }

    private func receive(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, self.running else { return }
            switch result {
            case .success(let msg):
                if case .string(let text) = msg { self.handle(text) }
                self.receive(task)
            case .failure:
                DispatchQueue.main.async { self.status = "connection lost — retrying…" }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    guard self.running, let url = task.originalRequest?.url else { return }
                    self.connect(url)
                }
            }
        }
    }

    private func handle(_ text: String) {
        guard let data = text.data(using: .utf8),
              let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              msg["type"] as? String == "matches",
              let matches = msg["matches"] as? [[String: Any]],
              let top = matches.first,
              let task = top["task"] as? String,
              let score = top["score"] as? Double else { return }
        DispatchQueue.main.async {
            self.status = "matching in \(msg["space"] as? String ?? "?") space"
            self.topMatch = "\(task.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " "))  ·  \(String(format: "%.2f", score))"
        }
    }
}
