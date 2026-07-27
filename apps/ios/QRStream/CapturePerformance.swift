import Foundation

public enum CameraCaptureTier: String, CaseIterable, Sendable {
    case ultraHD60
    case fullHD60
    case fullHD30

    public static let preferredOrder: [CameraCaptureTier] = [
        .ultraHD60,
        .fullHD60,
        .fullHD30,
    ]

    public var width: Int32 {
        switch self {
        case .ultraHD60:
            return 3840
        case .fullHD60, .fullHD30:
            return 1920
        }
    }

    public var height: Int32 {
        switch self {
        case .ultraHD60:
            return 2160
        case .fullHD60, .fullHD30:
            return 1080
        }
    }

    public var framesPerSecond: Int {
        switch self {
        case .ultraHD60, .fullHD60:
            return 60
        case .fullHD30:
            return 30
        }
    }

    public var frameBudgetMilliseconds: Double {
        1_000 / Double(framesPerSecond)
    }

    public var displayName: String {
        switch self {
        case .ultraHD60:
            return "4K @ 60"
        case .fullHD60:
            return "1080p @ 60"
        case .fullHD30:
            return "1080p @ 30"
        }
    }
}

enum CameraCaptureTierSelector {
    static func preferredTier(
        supportedTiers: Set<CameraCaptureTier>
    ) -> CameraCaptureTier? {
        CameraCaptureTier.preferredOrder.first { supportedTiers.contains($0) }
    }

    static func nextLowerTier(
        after activeTier: CameraCaptureTier,
        supportedTiers: Set<CameraCaptureTier>
    ) -> CameraCaptureTier? {
        guard let index = CameraCaptureTier.preferredOrder.firstIndex(of: activeTier)
        else {
            return preferredTier(supportedTiers: supportedTiers)
        }
        return CameraCaptureTier.preferredOrder
            .dropFirst(index + 1)
            .first { supportedTiers.contains($0) }
    }
}

public struct ScannerPerformanceSnapshot: Equatable, Sendable {
    public var activeTier: CameraCaptureTier?
    public var capturedWidth: Int
    public var capturedHeight: Int
    public var deliveredFrames: UInt64
    public var detectionAttempts: UInt64
    public var droppedFrames: UInt64
    public var deliveredFramesPerSecond: Double
    public var detectionAttemptsPerSecond: Double
    public var averageDetectionMilliseconds: Double
    public var p95DetectionMilliseconds: Double
    public var thermalState: String
    public var statusMessage: String?

    public init(
        activeTier: CameraCaptureTier? = nil,
        capturedWidth: Int = 0,
        capturedHeight: Int = 0,
        deliveredFrames: UInt64 = 0,
        detectionAttempts: UInt64 = 0,
        droppedFrames: UInt64 = 0,
        deliveredFramesPerSecond: Double = 0,
        detectionAttemptsPerSecond: Double = 0,
        averageDetectionMilliseconds: Double = 0,
        p95DetectionMilliseconds: Double = 0,
        thermalState: String = "unknown",
        statusMessage: String? = nil
    ) {
        self.activeTier = activeTier
        self.capturedWidth = capturedWidth
        self.capturedHeight = capturedHeight
        self.deliveredFrames = deliveredFrames
        self.detectionAttempts = detectionAttempts
        self.droppedFrames = droppedFrames
        self.deliveredFramesPerSecond = deliveredFramesPerSecond
        self.detectionAttemptsPerSecond = detectionAttemptsPerSecond
        self.averageDetectionMilliseconds = averageDetectionMilliseconds
        self.p95DetectionMilliseconds = p95DetectionMilliseconds
        self.thermalState = thermalState
        self.statusMessage = statusMessage
    }
}

struct BoundedDoubleWindow {
    private(set) var values: [Double] = []
    let capacity: Int

    init(capacity: Int) {
        precondition(capacity > 1)
        self.capacity = capacity
        values.reserveCapacity(capacity)
    }

    mutating func append(_ value: Double) {
        values.append(value)
        if values.count > capacity {
            values.removeFirst(values.count - capacity)
        }
    }

    mutating func reset() {
        values.removeAll(keepingCapacity: true)
    }

    var average: Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    func percentile(_ percentile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let bounded = min(max(percentile, 0), 1)
        let index = max(0, Int(ceil(bounded * Double(sorted.count))) - 1)
        return sorted[index]
    }

    var ratePerSecond: Double {
        guard values.count > 1,
              let first = values.first,
              let last = values.last,
              last > first
        else {
            return 0
        }
        return Double(values.count - 1) / (last - first)
    }
}

struct ScannerPerformanceAccumulator {
    private(set) var activeTier: CameraCaptureTier?
    private(set) var capturedWidth: Int = 0
    private(set) var capturedHeight: Int = 0
    private(set) var deliveredFrames: UInt64 = 0
    private(set) var detectionAttempts: UInt64 = 0
    private(set) var droppedFrames: UInt64 = 0

    private var presentationTimestamps = BoundedDoubleWindow(capacity: 240)
    private var detectionTimestamps = BoundedDoubleWindow(capacity: 240)
    private var detectionLatencies = BoundedDoubleWindow(capacity: 240)
    private var lastEvaluationAttempt: UInt64 = 0
    private var consecutiveSlowWindows = 0

    mutating func reset(activeTier: CameraCaptureTier?) {
        self.activeTier = activeTier
        capturedWidth = 0
        capturedHeight = 0
        deliveredFrames = 0
        detectionAttempts = 0
        droppedFrames = 0
        presentationTimestamps.reset()
        detectionTimestamps.reset()
        detectionLatencies.reset()
        lastEvaluationAttempt = 0
        consecutiveSlowWindows = 0
    }

    mutating func recordDeliveredFrame(
        presentationTimestampSeconds: Double,
        width: Int = 0,
        height: Int = 0
    ) {
        deliveredFrames += 1
        capturedWidth = width
        capturedHeight = height
        presentationTimestamps.append(presentationTimestampSeconds)
    }

    mutating func recordDetection(
        latencyMilliseconds: Double,
        wallTimeSeconds: Double
    ) {
        detectionAttempts += 1
        detectionLatencies.append(latencyMilliseconds)
        detectionTimestamps.append(wallTimeSeconds)
    }

    mutating func recordDroppedFrame() {
        droppedFrames += 1
    }

    func snapshot(
        thermalState: String,
        statusMessage: String? = nil
    ) -> ScannerPerformanceSnapshot {
        ScannerPerformanceSnapshot(
            activeTier: activeTier,
            capturedWidth: capturedWidth,
            capturedHeight: capturedHeight,
            deliveredFrames: deliveredFrames,
            detectionAttempts: detectionAttempts,
            droppedFrames: droppedFrames,
            deliveredFramesPerSecond: presentationTimestamps.ratePerSecond,
            detectionAttemptsPerSecond: detectionTimestamps.ratePerSecond,
            averageDetectionMilliseconds: detectionLatencies.average,
            p95DetectionMilliseconds: detectionLatencies.percentile(0.95),
            thermalState: thermalState,
            statusMessage: statusMessage
        )
    }

    mutating func sustainedOverloadReason() -> String? {
        guard let activeTier else { return nil }

        let warmupFrames = UInt64(activeTier.framesPerSecond)
        guard detectionAttempts >= warmupFrames else { return nil }

        if droppedFrames > 0 {
            return "\(droppedFrames) capture frame(s) dropped"
        }

        let evaluationInterval = UInt64(activeTier.framesPerSecond)
        guard detectionAttempts - lastEvaluationAttempt >= evaluationInterval
        else {
            return nil
        }
        lastEvaluationAttempt = detectionAttempts

        let p95 = detectionLatencies.percentile(0.95)
        let detectionRate = detectionTimestamps.ratePerSecond
        let minimumRate = Double(activeTier.framesPerSecond) * 0.98
        let isSlow = p95 > activeTier.frameBudgetMilliseconds
            || (detectionRate > 0 && detectionRate < minimumRate)

        consecutiveSlowWindows = isSlow ? consecutiveSlowWindows + 1 : 0
        guard consecutiveSlowWindows >= 2 else { return nil }

        if p95 > activeTier.frameBudgetMilliseconds {
            return String(
                format: "detector P95 %.2f ms exceeds %.2f ms budget",
                p95,
                activeTier.frameBudgetMilliseconds
            )
        }
        return String(
            format: "detector rate %.2f fps is below %.2f fps target",
            detectionRate,
            minimumRate
        )
    }
}
