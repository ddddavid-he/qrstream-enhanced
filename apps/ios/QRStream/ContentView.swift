import SwiftUI

/// Minimal application shell for the capture and decode pipeline.
///
/// Product UI intentionally lives outside this view so it can be rebuilt
/// without coupling visual components to the scanner implementation.
public struct ContentView: View {
    @EnvironmentObject private var decodeModel: DecodeSessionModel

    public init() {}

    public var body: some View {
        ScannerView { qrText in
            decodeModel.consume(qrText: qrText)
        }
        .ignoresSafeArea()
    }
}
