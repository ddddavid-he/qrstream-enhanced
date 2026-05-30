import SwiftUI

public struct DecodeProgressView: View {
    public let snapshot: DecodeSnapshot

    public init(snapshot: DecodeSnapshot) {
        self.snapshot = snapshot
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(snapshot.done ? "Complete" : "Receiving")
                    .font(.headline)
                Spacer()
                Text(progressText)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            ProgressView(value: snapshot.progress)

            Text(detailText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var progressText: String {
        "\(Int((snapshot.progress * 100).rounded()))%"
    }

    private var detailText: String {
        if !snapshot.initialized {
            return "Point the camera at a QRStream V4 stream."
        }
        return "\(snapshot.numRecovered)/\(snapshot.symbolCount) symbols"
    }
}

