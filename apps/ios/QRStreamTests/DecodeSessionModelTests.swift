import XCTest
@testable import QRStream

final class DecodeSessionModelTests: XCTestCase {
    func testCaptureTierSelectorUsesResolutionFirstFallbackOrder() {
        let allTiers = Set(CameraCaptureTier.allCases)

        XCTAssertEqual(
            CameraCaptureTierSelector.preferredTier(supportedTiers: allTiers),
            .ultraHD60
        )
        XCTAssertEqual(
            CameraCaptureTierSelector.nextLowerTier(
                after: .ultraHD60,
                supportedTiers: allTiers
            ),
            .fullHD60
        )
        XCTAssertEqual(
            CameraCaptureTierSelector.nextLowerTier(
                after: .fullHD60,
                supportedTiers: allTiers
            ),
            .fullHD30
        )
    }

    func testCaptureTierSelectorSkipsUnsupportedTiers() {
        let supported: Set<CameraCaptureTier> = [.fullHD30]

        XCTAssertEqual(
            CameraCaptureTierSelector.preferredTier(supportedTiers: supported),
            .fullHD30
        )
        XCTAssertEqual(
            CameraCaptureTierSelector.nextLowerTier(
                after: .ultraHD60,
                supportedTiers: supported
            ),
            .fullHD30
        )
        XCTAssertNil(
            CameraCaptureTierSelector.nextLowerTier(
                after: .fullHD30,
                supportedTiers: supported
            )
        )
    }

    func testPerformanceAccumulatorRequiresSustainedSlowWindows() {
        var accumulator = ScannerPerformanceAccumulator()
        accumulator.reset(activeTier: .ultraHD60)

        for frame in 0..<60 {
            accumulator.recordDeliveredFrame(
                presentationTimestampSeconds: Double(frame) / 60
            )
            accumulator.recordDetection(
                latencyMilliseconds: 20,
                wallTimeSeconds: Double(frame) / 60
            )
        }
        XCTAssertNil(accumulator.sustainedOverloadReason())

        for frame in 60..<120 {
            accumulator.recordDeliveredFrame(
                presentationTimestampSeconds: Double(frame) / 60
            )
            accumulator.recordDetection(
                latencyMilliseconds: 20,
                wallTimeSeconds: Double(frame) / 60
            )
        }
        XCTAssertNotNil(accumulator.sustainedOverloadReason())
    }

    func testPerformanceAccumulatorTreatsDropAsOverload() {
        var accumulator = ScannerPerformanceAccumulator()
        accumulator.reset(activeTier: .fullHD60)

        for frame in 0..<60 {
            accumulator.recordDeliveredFrame(
                presentationTimestampSeconds: Double(frame) / 60
            )
            accumulator.recordDetection(
                latencyMilliseconds: 4,
                wallTimeSeconds: Double(frame) / 60
            )
        }
        accumulator.recordDroppedFrame()

        XCTAssertEqual(
            accumulator.sustainedOverloadReason(),
            "1 capture frame(s) dropped"
        )
    }

    func testPlaceholderSessionAcceptsPayloadsAndUpdatesProgress() throws {
        let session = PlaceholderDecodeSession()

        let first = session.consumeQrText("qrstream-frame-1")
        let snapshot = session.snapshot()

        XCTAssertTrue(first.accepted)
        XCTAssertTrue(snapshot.initialized)
        XCTAssertEqual(snapshot.numRecovered, 1)
        XCTAssertEqual(snapshot.symbolCount, 100)
        XCTAssertEqual(snapshot.progress, 0.01)
    }

    func testPlaceholderSessionRejectsEmptyPayload() throws {
        let session = PlaceholderDecodeSession()

        let result = session.consumeQrText("")

        XCTAssertFalse(result.accepted)
        XCTAssertEqual(result.errorMessage, "empty QR payload")
        XCTAssertFalse(session.snapshot().initialized)
    }

    @MainActor
    func testDecodeSessionModelResetsState() throws {
        let model = DecodeSessionModel()

        model.consume(qrText: "qrstream-frame-1")
        model.reset()

        XCTAssertFalse(model.snapshot.initialized)
        XCTAssertEqual(model.snapshot.progress, 0)
    }
}
