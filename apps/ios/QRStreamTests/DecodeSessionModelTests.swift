import XCTest
@testable import QRStream

final class DecodeSessionModelTests: XCTestCase {
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
