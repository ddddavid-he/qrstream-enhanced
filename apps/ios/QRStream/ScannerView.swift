import SwiftUI

#if canImport(UIKit) && canImport(AVFoundation) && canImport(ZXingCpp)
import AVFoundation
import UIKit
import ZXingCpp

public struct ScannerView: UIViewControllerRepresentable {
    public let recognitionEnabled: Bool
    public let onPerformanceUpdate: (ScannerPerformanceSnapshot) -> Void
    public let onQRCode: (String) -> Void

    public init(
        recognitionEnabled: Bool = true,
        onPerformanceUpdate: @escaping (ScannerPerformanceSnapshot) -> Void = { _ in },
        onQRCode: @escaping (String) -> Void
    ) {
        self.recognitionEnabled = recognitionEnabled
        self.onPerformanceUpdate = onPerformanceUpdate
        self.onQRCode = onQRCode
    }

    public func makeUIViewController(context: Context) -> ScannerViewController {
        let controller = ScannerViewController()
        controller.onQRCode = onQRCode
        controller.onPerformanceUpdate = onPerformanceUpdate
        controller.setRecognitionEnabled(recognitionEnabled)
        return controller
    }

    public func updateUIViewController(
        _ uiViewController: ScannerViewController,
        context: Context
    ) {
        uiViewController.onQRCode = onQRCode
        uiViewController.onPerformanceUpdate = onPerformanceUpdate
        uiViewController.setRecognitionEnabled(recognitionEnabled)
    }
}

public final class ScannerViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    public var onQRCode: ((String) -> Void)? {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return qrCodeCallback
        }
        set {
            stateLock.lock()
            qrCodeCallback = newValue
            stateLock.unlock()
        }
    }
    public var onPerformanceUpdate: ((ScannerPerformanceSnapshot) -> Void)? {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return performanceCallback
        }
        set {
            stateLock.lock()
            performanceCallback = newValue
            stateLock.unlock()
        }
    }

    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var captureDevice: AVCaptureDevice?
    private var resolvedFormats: [CameraCaptureTier: AVCaptureDevice.Format] = [:]
    private var activeTier: CameraCaptureTier?

    /// Session configuration and start/stop can block, so it must not share the
    /// main or detector queue.
    private let sessionQueue = DispatchQueue(
        label: "dev.qrstream.scanner.session",
        qos: .userInitiated
    )

    /// Serial delivery guarantees that each delivered frame is detected in PTS
    /// order. AVFoundation drop callbacks expose overload instead of silently
    /// hiding it.
    private let detectionQueue = DispatchQueue(
        label: "dev.qrstream.scanner.detection",
        qos: .userInitiated
    )

    private let payloadDeduper = PayloadDeduper()
    private let stateLock = NSLock()
    private var qrCodeCallback: ((String) -> Void)?
    private var performanceCallback: ((ScannerPerformanceSnapshot) -> Void)?
    private var recognitionEnabled = true
    private var currentTierForCallbacks: CameraCaptureTier?

    /// Accessed only on `detectionQueue`.
    private var performance = ScannerPerformanceAccumulator()
    private var lastMetricsPublishTime: Double = 0
    private var performanceStatusMessage: String?
    private var downgradePending = false

    /// Reused across frames so ZXingCpp does not pay reader setup costs on each
    /// sample. A miss is discarded immediately; hard-mode retry is forbidden
    /// because it blocks the serial detector queue and destroys live FPS.
    private lazy var reader = makeReader()

    private func makeReader() -> ZXIBarcodeReader {
        let options = ZXIReaderOptions()
        options.formats = [NSNumber(value: ZXIFormat.QR_CODE.rawValue)]
        options.tryHarder = false
        options.tryRotate = false
        options.tryInvert = false
        options.tryDownscale = true
        options.maxNumberOfSymbols = 4
        return ZXIBarcodeReader(options: options)
    }

    public override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }

    public override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    public override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        sessionQueue.async { [session] in
            if !session.isRunning {
                session.startRunning()
            }
        }
    }

    public override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    public func setRecognitionEnabled(_ enabled: Bool) {
        stateLock.lock()
        let changed = recognitionEnabled != enabled
        recognitionEnabled = enabled
        let tier = currentTierForCallbacks
        stateLock.unlock()
        guard changed else { return }

        sessionQueue.async { [weak self] in
            self?.videoOutput?.connection(with: .video)?.isEnabled = enabled
        }

        if enabled {
            payloadDeduper.reset()
            detectionQueue.async { [weak self] in
                guard let self else { return }
                self.performance.reset(activeTier: tier)
                self.lastMetricsPublishTime = 0
                self.publishPerformance(statusMessage: "Recognition restarted")
            }
        }
    }

    private func configureSession() {
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            reportConfigurationFailure("Back camera unavailable")
            return
        }

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            reportConfigurationFailure("Camera input unavailable: \(error.localizedDescription)")
            return
        }

        session.beginConfiguration()
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            reportConfigurationFailure("Camera input cannot be added")
            return
        }
        session.addInput(input)
        if session.canSetSessionPreset(.inputPriority) {
            session.sessionPreset = .inputPriority
        }

        resolvedFormats = resolveCaptureFormats(for: device)
        let supportedTiers = Set(resolvedFormats.keys)
        guard let initialTier = CameraCaptureTierSelector.preferredTier(
            supportedTiers: supportedTiers
        ), let initialFormat = resolvedFormats[initialTier] else {
            session.commitConfiguration()
            reportConfigurationFailure("Device does not support 1080p @ 30 or better")
            return
        }

        do {
            try apply(format: initialFormat, tier: initialTier, to: device)
        } catch {
            session.commitConfiguration()
            reportConfigurationFailure(
                "Could not configure \(initialTier.displayName): \(error.localizedDescription)"
            )
            return
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = false
        let pixelFormat = preferredNV12PixelFormat(for: videoOutput)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat
        ]
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            reportConfigurationFailure("Camera video output cannot be added")
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
            connection.isEnabled = isRecognitionEnabled()
        }

        session.commitConfiguration()
        captureDevice = device
        self.videoOutput = videoOutput
        activeTier = initialTier
        setCurrentTierForCallbacks(initialTier)

        detectionQueue.async { [weak self] in
            guard let self else { return }
            self.performance.reset(activeTier: initialTier)
            self.publishPerformance(
                statusMessage: "Selected \(initialTier.displayName) · NV12"
            )
        }

        DispatchQueue.main.async { [weak self, session] in
            guard let self else { return }
            let layer = AVCaptureVideoPreviewLayer(session: session)
            layer.videoGravity = .resizeAspectFill
            self.view.layer.insertSublayer(layer, at: 0)
            self.previewLayer = layer
            layer.frame = self.view.bounds
        }
    }

    private func resolveCaptureFormats(
        for device: AVCaptureDevice
    ) -> [CameraCaptureTier: AVCaptureDevice.Format] {
        var result: [CameraCaptureTier: AVCaptureDevice.Format] = [:]

        for tier in CameraCaptureTier.preferredOrder {
            let candidates = device.formats.filter { format in
                let dimensions = CMVideoFormatDescriptionGetDimensions(
                    format.formatDescription
                )
                guard dimensions.width == tier.width,
                      dimensions.height == tier.height
                else {
                    return false
                }
                return format.videoSupportedFrameRateRanges.contains { range in
                    range.minFrameRate <= Double(tier.framesPerSecond) + 0.01
                        && range.maxFrameRate >= Double(tier.framesPerSecond) - 0.5
                }
            }

            result[tier] = candidates.max { lhs, rhs in
                captureFormatScore(lhs) < captureFormatScore(rhs)
            }
        }
        return result
    }

    private func captureFormatScore(_ format: AVCaptureDevice.Format) -> Int {
        let subtype = CMFormatDescriptionGetMediaSubType(format.formatDescription)
        switch subtype {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return 2
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            return 1
        default:
            return 0
        }
    }

    private func preferredNV12PixelFormat(
        for output: AVCaptureVideoDataOutput
    ) -> OSType {
        if output.availableVideoPixelFormatTypes.contains(
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ) {
            return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        }
        return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }

    private func apply(
        format: AVCaptureDevice.Format,
        tier: CameraCaptureTier,
        to device: AVCaptureDevice
    ) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        device.activeFormat = format
        let duration = CMTime(
            value: 1,
            timescale: CMTimeScale(tier.framesPerSecond)
        )
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
    }

    private func requestDowngrade(reason: String) {
        guard !downgradePending else { return }
        downgradePending = true
        sessionQueue.async { [weak self] in
            self?.applyNextCaptureTier(reason: reason)
        }
    }

    private func applyNextCaptureTier(reason: String) {
        guard let device = captureDevice,
              let activeTier
        else {
            finishDowngrade(statusMessage: "Capture profile unavailable")
            return
        }

        let supportedTiers = Set(resolvedFormats.keys)
        guard let nextTier = CameraCaptureTierSelector.nextLowerTier(
            after: activeTier,
            supportedTiers: supportedTiers
        ), let nextFormat = resolvedFormats[nextTier] else {
            finishDowngrade(
                statusMessage: "\(activeTier.displayName) overloaded: \(reason); no lower SLA tier"
            )
            return
        }

        let connection = videoOutput?.connection(with: .video)
        connection?.isEnabled = false
        detectionQueue.sync {}

        do {
            try apply(format: nextFormat, tier: nextTier, to: device)
            self.activeTier = nextTier
            setCurrentTierForCallbacks(nextTier)
            detectionQueue.sync { [weak self] in
                guard let self else { return }
                self.performance.reset(activeTier: nextTier)
                self.lastMetricsPublishTime = 0
            }
            connection?.isEnabled = isRecognitionEnabled()
            finishDowngrade(
                statusMessage: "Downgraded \(activeTier.displayName) → \(nextTier.displayName): \(reason)"
            )
        } catch {
            connection?.isEnabled = isRecognitionEnabled()
            finishDowngrade(
                statusMessage: "Could not downgrade to \(nextTier.displayName): \(error.localizedDescription)"
            )
        }
    }

    private func finishDowngrade(statusMessage: String) {
        detectionQueue.async { [weak self] in
            guard let self else { return }
            self.downgradePending = false
            self.publishPerformance(statusMessage: statusMessage)
        }
    }

    private func setCurrentTierForCallbacks(_ tier: CameraCaptureTier?) {
        stateLock.lock()
        currentTierForCallbacks = tier
        stateLock.unlock()
    }

    private func isRecognitionEnabled() -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return recognitionEnabled
    }

    private func reportConfigurationFailure(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.showUnavailableLabel(message)
            self?.onPerformanceUpdate?(
                ScannerPerformanceSnapshot(statusMessage: message)
            )
        }
    }

    private func showUnavailableLabel(_ message: String) {
        let label = UILabel()
        label.text = message
        label.textColor = .white
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            label.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
        ])
    }

    private func publishPerformance(statusMessage: String? = nil) {
        if let statusMessage {
            performanceStatusMessage = statusMessage
        }
        let snapshot = performance.snapshot(
            thermalState: currentThermalState(),
            statusMessage: performanceStatusMessage
        )
        let callback = onPerformanceUpdate
        DispatchQueue.main.async {
            callback?(snapshot)
        }
    }

    private func currentThermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return "nominal"
        case .fair:
            return "fair"
        case .serious:
            return "serious"
        case .critical:
            return "critical"
        @unknown default:
            return "unknown"
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard isRecognitionEnabled(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return
        }

        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        performance.recordDeliveredFrame(
            presentationTimestampSeconds: CMTimeGetSeconds(presentationTime),
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer)
        )

        let start = ProcessInfo.processInfo.systemUptime
        var results: [ZXIResult] = []
        do {
            results = try reader.read(pixelBuffer)
        } catch {
            results = []
        }
        let end = ProcessInfo.processInfo.systemUptime
        performance.recordDetection(
            latencyMilliseconds: (end - start) * 1_000,
            wallTimeSeconds: end
        )

        if end - lastMetricsPublishTime >= 1 {
            lastMetricsPublishTime = end
            publishPerformance()
        }

        if let reason = performance.sustainedOverloadReason() {
            requestDowngrade(reason: reason)
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

    public func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard isRecognitionEnabled() else { return }
        performance.recordDroppedFrame()
        if let reason = performance.sustainedOverloadReason() {
            requestDowngrade(reason: reason)
        }
        publishPerformance()
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
    public let recognitionEnabled: Bool
    public let onPerformanceUpdate: (ScannerPerformanceSnapshot) -> Void
    public let onQRCode: (String) -> Void

    public init(
        recognitionEnabled: Bool = true,
        onPerformanceUpdate: @escaping (ScannerPerformanceSnapshot) -> Void = { _ in },
        onQRCode: @escaping (String) -> Void
    ) {
        self.recognitionEnabled = recognitionEnabled
        self.onPerformanceUpdate = onPerformanceUpdate
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
