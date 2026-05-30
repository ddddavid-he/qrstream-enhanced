import SwiftUI

public struct ContentView: View {
    @EnvironmentObject private var decodeModel: DecodeSessionModel

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                ScannerView { text in
                    decodeModel.consume(qrText: text)
                }
                .frame(maxWidth: .infinity, minHeight: 360)
                .clipShape(RoundedRectangle(cornerRadius: 20))

                DecodeProgressView(snapshot: decodeModel.snapshot)

                if let message = decodeModel.statusMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("QRStream")
            .toolbar {
                Button("Reset") {
                    decodeModel.reset()
                }
            }
        }
    }
}

