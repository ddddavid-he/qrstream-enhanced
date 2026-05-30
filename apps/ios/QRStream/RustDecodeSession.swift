import Foundation

#if canImport(qrstreamcoreFFI)
public final class RustDecodeSession: QRStreamDecodeSession {
    private let session: FfiV4DecodeSession

    public init() {
        self.session = FfiV4DecodeSession()
    }

    public func consumeQrText(_ text: String) -> DecodeResult {
        convert(session.consumeQrText(qrText: text))
    }

    public func snapshot() -> DecodeSnapshot {
        convert(session.snapshot())
    }

    public func resultBytes() throws -> Data {
        let data = session.resultBytes()
        guard !data.isEmpty || snapshot().done else {
            throw DecodeSessionError.incomplete
        }
        return data
    }

    public func reset() {
        session.reset()
    }

    private func convert(_ result: FfiDecodeResult) -> DecodeResult {
        DecodeResult(
            accepted: result.accepted,
            duplicate: result.duplicate,
            done: result.done,
            progress: result.progress,
            numRecovered: result.numRecovered,
            symbolCount: result.symbolCount,
            filesize: result.filesize,
            errorMessage: result.errorMessage
        )
    }

    private func convert(_ snapshot: FfiDecodeSnapshot) -> DecodeSnapshot {
        DecodeSnapshot(
            initialized: snapshot.initialized,
            done: snapshot.done,
            progress: snapshot.progress,
            numRecovered: snapshot.numRecovered,
            symbolCount: snapshot.symbolCount,
            filesize: snapshot.filesize
        )
    }
}
#endif
