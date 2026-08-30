import SwiftUI

struct ContentView: View {
    @StateObject private var streamer = MotionStreamer()
    @AppStorage("server") private var server = "192.168.1.10:7860"

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                Text("MUSCLE MEMORY")
                    .font(.system(size: 13, weight: .bold))
                    .kerning(1.5)

                TextField("laptop-ip:7860", text: $server)
                    .font(.system(size: 13, design: .monospaced))
                    .disabled(streamer.running)

                Button(streamer.running ? "Stop" : "Stream wrist motion") {
                    streamer.running ? streamer.stop() : streamer.start(host: server)
                }
                .tint(streamer.running ? .red : .blue)

                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4).fill(.quaternary)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.blue)
                            .frame(width: geo.size.width * streamer.energy)
                    }
                }
                .frame(height: 8)
                .animation(.linear(duration: 0.1), value: streamer.energy)

                if !streamer.topMatch.isEmpty {
                    Text(streamer.topMatch)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.center)
                }

                Text(streamer.status)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
    }
}

#Preview {
    ContentView()
}
