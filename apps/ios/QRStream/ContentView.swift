import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

public struct ContentView: View {
    @EnvironmentObject private var decodeModel: DecodeSessionModel
    @State private var shareURL: URL?
    @State private var shareError: String?
    @State private var scannerPerformance = ScannerPerformanceSnapshot()
    @State private var showScannerDebug = false
    @State private var isScanning = false

    public init() {}

    public var body: some View {
        ZStack {
            ScannerView(
                recognitionEnabled: isScanning && !decodeModel.snapshot.done,
                onPerformanceUpdate: { scannerPerformance = $0 },
                onQRCode: { text in
                    decodeModel.consume(qrText: text)
                }
            )
            .ignoresSafeArea()

            LinearGradient(
                colors: [
                    .black.opacity(0.52),
                    .clear,
                    .clear,
                    .black.opacity(0.72),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 14) {
                cameraToolbar

                if showScannerDebug {
                    ScannerPerformanceDetailsView(snapshot: scannerPerformance)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                Spacer(minLength: 16)

                ScannerCrosshair(isActive: isScanning)
                    .frame(width: 58, height: 58)

                Spacer(minLength: 16)

                CameraProgressOverlay(
                    snapshot: decodeModel.snapshot,
                    isScanning: isScanning,
                    statusMessage: decodeModel.statusMessage,
                    errorMessage: shareError
                )

                cameraControls
            }
            .padding(.horizontal, 18)
            .safeAreaPadding(.top, 8)
            .safeAreaPadding(.bottom, 10)
        }
        .background(.black)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.2), value: showScannerDebug)
        .onChange(of: decodeModel.snapshot.done) { _, isDone in
            if isDone {
                isScanning = false
            }
        }
        #if canImport(UIKit)
        .sheet(item: $shareURL) { url in
            ActivityView(activityItems: [url])
        }
        #endif
    }

    private var cameraToolbar: some View {
        HStack(spacing: 12) {
            Text("QRStream")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.white)

            Spacer()

            CameraToolbarButton(
                systemName: showScannerDebug ? "waveform.path.ecg.rectangle.fill" : "waveform.path.ecg.rectangle",
                accessibilityLabel: showScannerDebug ? "Hide debug metrics" : "Show debug metrics"
            ) {
                showScannerDebug.toggle()
            }

            CameraToolbarButton(
                systemName: "arrow.counterclockwise",
                accessibilityLabel: "Reset receiving session"
            ) {
                resetSession()
            }
        }
    }

    private var cameraControls: some View {
        HStack(alignment: .center) {
            Color.clear
                .frame(width: 72, height: 56)

            Spacer()

            CameraScanButton(
                isScanning: isScanning,
                isDisabled: decodeModel.snapshot.done
            ) {
                isScanning.toggle()
            }

            Spacer()

            if decodeModel.snapshot.done {
                CameraToolbarButton(
                    systemName: "square.and.arrow.up",
                    accessibilityLabel: "Share decoded file",
                    size: 56
                ) {
                    prepareShare()
                }
                .frame(width: 72, height: 56)
            } else {
                Color.clear
                    .frame(width: 72, height: 56)
            }
        }
    }

    private func resetSession() {
        isScanning = false
        decodeModel.reset()
        shareURL = nil
        shareError = nil
    }

    private func prepareShare() {
        do {
            let data = try decodeModel.resultBytes()
            let ext = sniffExtension(data)
            let filename = "qrstream-\(Int(Date().timeIntervalSince1970)).\(ext)"
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
            try data.write(to: url, options: .atomic)
            shareURL = url
            shareError = nil
        } catch {
            shareError = "Could not prepare file: \(error.localizedDescription)"
        }
    }

    private func sniffExtension(_ data: Data) -> String {
        // Magic-byte sniff covering common types we encode through QRStream.
        guard data.count >= 4 else { return "bin" }
        let bytes = [UInt8](data.prefix(16))

        if bytes.starts(with: [0x25, 0x50, 0x44, 0x46]) { return "pdf" }                         // %PDF
        if bytes.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return "png" }                         // PNG
        if bytes.starts(with: [0xFF, 0xD8, 0xFF]) { return "jpg" }                               // JPEG
        if bytes.starts(with: [0x47, 0x49, 0x46, 0x38]) { return "gif" }                         // GIF8
        if bytes.starts(with: [0x50, 0x4B, 0x03, 0x04]) { return "zip" }                         // PK..
        if bytes.starts(with: [0x1F, 0x8B]) { return "gz" }                                      // gzip
        if bytes.count >= 8 && Array(bytes[4..<8]) == [0x66, 0x74, 0x79, 0x70] { return "mp4" }  // ftyp
        // UTF-8 plain text heuristic: no NUL in the first chunk.
        if !data.prefix(512).contains(0x00),
           String(data: data.prefix(512), encoding: .utf8) != nil {
            return "txt"
        }
        return "bin"
    }
}

private struct CameraToolbarButton: View {
    let systemName: String
    let accessibilityLabel: String
    var size: CGFloat = 44
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: size * 0.4, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: size, height: size)
                .background(.black.opacity(0.46), in: Circle())
                .overlay {
                    Circle()
                        .stroke(.white.opacity(0.16), lineWidth: 1)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

private struct CameraScanButton: View {
    let isScanning: Bool
    let isDisabled: Bool
    let action: () -> Void

    var body: some View {
        VStack(spacing: 7) {
            Button(action: action) {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 78, height: 78)

                    if isScanning {
                        RoundedRectangle(cornerRadius: 7)
                            .fill(.red)
                            .frame(width: 30, height: 30)
                    } else {
                        Circle()
                            .fill(isDisabled ? .gray : .red)
                            .frame(width: 62, height: 62)
                    }
                }
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(isDisabled)
            .accessibilityLabel(isScanning ? "Stop scanning" : "Start scanning")

            Text(buttonCaption)
                .font(.caption.weight(.medium))
                .foregroundStyle(.white.opacity(isDisabled ? 0.65 : 0.92))
        }
        .frame(width: 112)
    }

    private var buttonCaption: String {
        if isDisabled {
            return "Complete"
        }
        return isScanning ? "Stop" : "Scan"
    }
}

private struct ScannerCrosshair: View {
    let isActive: Bool

    var body: some View {
        CameraCrosshairShape()
            .stroke(
                isActive ? Color.yellow : Color.white,
                style: StrokeStyle(lineWidth: 2.5, lineCap: .round)
            )
            .shadow(color: .black.opacity(0.65), radius: 3, y: 1)
        .accessibilityHidden(true)
    }
}

private struct CameraCrosshairShape: Shape {
    func path(in rect: CGRect) -> Path {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) * 0.14
        let gap = radius + 5
        let outerInset = min(rect.width, rect.height) * 0.08
        var path = Path()

        path.addEllipse(
            in: CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: radius * 2,
                height: radius * 2
            )
        )

        path.move(to: CGPoint(x: center.x, y: rect.minY + outerInset))
        path.addLine(to: CGPoint(x: center.x, y: center.y - gap))

        path.move(to: CGPoint(x: center.x, y: center.y + gap))
        path.addLine(to: CGPoint(x: center.x, y: rect.maxY - outerInset))

        path.move(to: CGPoint(x: rect.minX + outerInset, y: center.y))
        path.addLine(to: CGPoint(x: center.x - gap, y: center.y))

        path.move(to: CGPoint(x: center.x + gap, y: center.y))
        path.addLine(to: CGPoint(x: rect.maxX - outerInset, y: center.y))

        return path
    }
}

private struct CameraProgressOverlay: View {
    let snapshot: DecodeSnapshot
    let isScanning: Bool
    let statusMessage: String?
    let errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: iconName)
                    .font(.headline)
                    .foregroundStyle(.white)

                Spacer()

                Text(progressText)
                    .font(.headline.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.82))
            }

            ProgressView(value: snapshot.progress)
                .tint(snapshot.done ? .green : .yellow)

            Text(detailText)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.76))

            if let statusMessage {
                Text(statusMessage)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.76))
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(14)
        .background(.black.opacity(0.48), in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            RoundedRectangle(cornerRadius: 18)
                .stroke(.white.opacity(0.12), lineWidth: 1)
        }
    }

    private var title: String {
        if snapshot.done {
            return "Complete"
        }
        return isScanning ? "Scanning" : "Ready"
    }

    private var iconName: String {
        if snapshot.done {
            return "checkmark.circle.fill"
        }
        return isScanning ? "dot.radiowaves.left.and.right" : "viewfinder"
    }

    private var progressText: String {
        "\(Int((snapshot.progress * 100).rounded()))%"
    }

    private var detailText: String {
        if snapshot.done {
            return "\(snapshot.numRecovered)/\(snapshot.symbolCount) symbols received"
        }
        if snapshot.initialized {
            return "\(snapshot.numRecovered)/\(snapshot.symbolCount) symbols received"
        }
        return isScanning
            ? "Scanning the entire camera frame."
            : "Preview is active. Tap Scan to recognize the entire frame."
    }
}

private struct ScannerPerformanceDetailsView: View {
    let snapshot: ScannerPerformanceSnapshot

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            metricRow(
                "Capture",
                captureDescription
            )
            metricRow(
                "Delivered / detect",
                String(
                    format: "%.1f / %.1f fps",
                    snapshot.deliveredFramesPerSecond,
                    snapshot.detectionAttemptsPerSecond
                )
            )
            metricRow(
                "ZXing avg / P95",
                String(
                    format: "%.2f / %.2f ms",
                    snapshot.averageDetectionMilliseconds,
                    snapshot.p95DetectionMilliseconds
                )
            )
            metricRow(
                "Frames / drops",
                "\(snapshot.detectionAttempts) / \(snapshot.droppedFrames)"
            )
            metricRow("Thermal", snapshot.thermalState)
            if let statusMessage = snapshot.statusMessage {
                metricRow("Status", statusMessage)
            }
        }
        .font(.caption.monospacedDigit())
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .foregroundStyle(.white)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(.white.opacity(0.12), lineWidth: 1)
        }
    }

    private var captureDescription: String {
        guard let tier = snapshot.activeTier else { return "Unavailable" }
        guard snapshot.capturedWidth > 0, snapshot.capturedHeight > 0 else {
            return tier.displayName
        }
        return "\(tier.displayName) · \(snapshot.capturedWidth)×\(snapshot.capturedHeight)"
    }

    @ViewBuilder
    private func metricRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(.white.opacity(0.62))
            Text(value)
                .textSelection(.enabled)
        }
    }
}

#if canImport(UIKit)
extension URL: Identifiable {
    public var id: String { absoluteString }
}

private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
