import Foundation

public struct DecodeSnapshot: Equatable {
    public var initialized: Bool
    public var done: Bool
    public var progress: Double
    public var numRecovered: UInt64
    public var symbolCount: UInt64
    public var filesize: UInt64

    public init(
        initialized: Bool = false,
        done: Bool = false,
        progress: Double = 0,
        numRecovered: UInt64 = 0,
        symbolCount: UInt64 = 0,
        filesize: UInt64 = 0
    ) {
        self.initialized = initialized
        self.done = done
        self.progress = progress
        self.numRecovered = numRecovered
        self.symbolCount = symbolCount
        self.filesize = filesize
    }
}

public struct DecodeResult: Equatable {
    public var accepted: Bool
    public var duplicate: Bool
    public var done: Bool
    public var progress: Double
    public var numRecovered: UInt64
    public var symbolCount: UInt64
    public var filesize: UInt64
    public var errorMessage: String?

    public init(
        accepted: Bool,
        duplicate: Bool,
        done: Bool,
        progress: Double,
        numRecovered: UInt64,
        symbolCount: UInt64,
        filesize: UInt64,
        errorMessage: String? = nil
    ) {
        self.accepted = accepted
        self.duplicate = duplicate
        self.done = done
        self.progress = progress
        self.numRecovered = numRecovered
        self.symbolCount = symbolCount
        self.filesize = filesize
        self.errorMessage = errorMessage
    }
}

public protocol QRStreamDecodeSession {
    func consumeQrText(_ text: String) -> DecodeResult
    func snapshot() -> DecodeSnapshot
    func resultBytes() throws -> Data
    func reset()
}

public enum DecodeSessionError: Error, Equatable {
    case incomplete
}

public final class PlaceholderDecodeSession: QRStreamDecodeSession {
    private var latestSnapshot = DecodeSnapshot()
    private var acceptedCount: UInt64 = 0

    public init() {}

    public func consumeQrText(_ text: String) -> DecodeResult {
        guard !text.isEmpty else {
            return DecodeResult(
                accepted: false,
                duplicate: false,
                done: latestSnapshot.done,
                progress: latestSnapshot.progress,
                numRecovered: latestSnapshot.numRecovered,
                symbolCount: latestSnapshot.symbolCount,
                filesize: latestSnapshot.filesize,
                errorMessage: "empty QR payload"
            )
        }

        acceptedCount += 1
        let symbolCount = max(UInt64(100), latestSnapshot.symbolCount)
        let recovered = min(acceptedCount, symbolCount)
        latestSnapshot = DecodeSnapshot(
            initialized: true,
            done: recovered == symbolCount,
            progress: Double(recovered) / Double(symbolCount),
            numRecovered: recovered,
            symbolCount: symbolCount,
            filesize: latestSnapshot.filesize
        )

        return DecodeResult(
            accepted: true,
            duplicate: false,
            done: latestSnapshot.done,
            progress: latestSnapshot.progress,
            numRecovered: latestSnapshot.numRecovered,
            symbolCount: latestSnapshot.symbolCount,
            filesize: latestSnapshot.filesize
        )
    }

    public func snapshot() -> DecodeSnapshot {
        latestSnapshot
    }

    public func resultBytes() throws -> Data {
        guard latestSnapshot.done else {
            throw DecodeSessionError.incomplete
        }
        return Data()
    }

    public func reset() {
        latestSnapshot = DecodeSnapshot()
        acceptedCount = 0
    }
}

@MainActor
public final class DecodeSessionModel: ObservableObject {
    @Published public private(set) var snapshot = DecodeSnapshot()
    @Published public private(set) var statusMessage: String?

    private let session: QRStreamDecodeSession

    public init(session: QRStreamDecodeSession = PlaceholderDecodeSession()) {
        self.session = session
    }

    public func consume(qrText: String) {
        let result = session.consumeQrText(qrText)
        snapshot = session.snapshot()

        if result.done {
            statusMessage = "Decoded \(result.filesize) bytes"
        } else if result.duplicate {
            statusMessage = "Duplicate frame"
        } else if !result.accepted {
            statusMessage = result.errorMessage ?? "Ignored non-QRStream payload"
        } else {
            statusMessage = "Recovered \(result.numRecovered)/\(result.symbolCount) symbols"
        }
    }

    public func reset() {
        session.reset()
        snapshot = session.snapshot()
        statusMessage = nil
    }
}
