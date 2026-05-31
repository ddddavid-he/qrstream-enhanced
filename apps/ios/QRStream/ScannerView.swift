import SwiftUI

#if canImport(UIKit) && canImport(AVFoundation) && canImport(ZXingCpp)
import AVFoundation
import UIKit
import ZXingCpp

public struct ScannerView: UIViewControllerRepresentable {
    public let onQRCode: (String) -> Void

    public init(onQRCode: @escaping (String) -> Void) {
        self.onQRCode = onQRCode
    }

    public func makeUIViewController(context: Context) -> ScannerViewController {
        let controller = ScannerViewController()
        controller.onQRCode = onQRCode
        return controller
    }

    public func updateUIViewController(_ uiViewController: ScannerViewController, context: Context) {}
}

public final class ScannerViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    public var onQRCode: ((String) -> Void)?

    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?

    /// Dedicated queue for AVFoundation sample-buffer delivery and ZXingCpp decoding.
    /// Detection is non-trivially CPU-bound (~10-30 ms per 1080p frame with
    /// `tryHarder` on), so keeping it off the main queue prevents UI stalls.
    private let detectionQueue = DispatchQueue(
        label: "dev.qrstream.scanner.detection",
        qos: .userInitiated
    )

    private let payloadDeduper = PayloadDeduper()

    /// Reused across frames so ZXingCpp doesn't pay setup costs on every frame.
    /// `tryHarder` trades a few ms of latency for substantially higher
    /// recognition rate on dense (version 30+) symbols at >15 fps.
    private lazy var reader: ZXIBarcodeReader = {
        let options = ZXIReaderOptions()
        options.formats = [NSNumber(value: ZXIFormat.QR_CODE.rawValue)]
        options.tryHarder = true
        options.tryRotate = false
        options.tryInvert = false
        options.tryDownscale = true
        // `maxNumberOfSymbols == 0` means "unlimited" in zxing-cpp; cap it so a
        // single frame with multiple QRs doesn't run away.
        options.maxNumberOfSymbols = 4
        return ZXIBarcodeReader(options: options)
    }()

    public override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureSession()
    }

    public override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    public override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !session.isRunning {
            detectionQueue.async { [session] in
                session.startRunning()
            }
        }
    }

    public override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if session.isRunning {
            session.stopRunning()
        }
    }

    private func configureSession() {
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else {
            showUnavailableLabel()
            return
        }

        session.beginConfiguration()
        // 1080p strikes a balance between zxing-cpp decoding cost and symbol
        // legibility. 4K roughly doubles per-frame latency without a
        // recognition-rate win for typical hand-held framing.
        if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        } else {
            session.sessionPreset = .high
        }
        session.addInput(input)

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        // BGRA matches ZXingCpp's most-tested input path on iOS.
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            showUnavailableLabel()
            return
        }
        videoOutput.setSampleBufferDelegate(self, queue: detectionQueue)
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }

        session.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        previewLayer = layer
    }

    private func showUnavailableLabel() {
        let label = UILabel()
        label.text = "Camera unavailable"
        label.textColor = .white
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // ZXingCpp accesses the pixel buffer's base address directly. Calling
        // synchronously keeps the sample buffer alive for the duration of the
        // read and avoids use-after-free.
        let results: [ZXIResult]
        do {
            results = try reader.read(pixelBuffer)
        } catch {
            // ZXingCpp throws on malformed frames; ignore and wait for the next.
            return
        }

        guard !results.isEmpty else { return }

        for result in results
        where result.format == ZXIFormat.QR_CODE && !result.text.isEmpty {
            let payload = result.text
            guard payloadDeduper.shouldEmit(payload) else { continue }
            let onQRCode = self.onQRCode
            DispatchQueue.main.async {
                onQRCode?(payload)
            }
        }
    }
}

/// Drops payloads we have already forwarded to the decode session.
///
/// AVCaptureVideoDataOutput hands us every frame, so the same QR symbol can
/// arrive dozens of times in a row. The Rust decoder is itself idempotent, but
/// the FFI hop is non-trivial — skipping duplicates here saves work and keeps
/// the status UI stable.
private final class PayloadDeduper {
    private let lock = NSLock()
    private var seen: Set<String> = []
    private var ringBuffer: [String] = []
    private let capacity = 4096

    func shouldEmit(_ payload: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if seen.contains(payload) { return false }
        seen.insert(payload)
        ringBuffer.append(payload)
        if ringBuffer.count > capacity {
            let evicted = ringBuffer.removeFirst()
            seen.remove(evicted)
        }
        return true
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        seen.removeAll()
        ringBuffer.removeAll()
    }
}
#else
public struct ScannerView: View {
    public let onQRCode: (String) -> Void

    public init(onQRCode: @escaping (String) -> Void) {
        self.onQRCode = onQRCode
    }

    public var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "qrcode.viewfinder")
                .font(.system(size: 48))
            Text("QR scanner preview is available on iOS devices.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Simulate QR Frame") {
                onQRCode("placeholder")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.opacity(0.08))
    }
}
#endif
