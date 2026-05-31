import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

public struct ContentView: View {
    @EnvironmentObject private var decodeModel: DecodeSessionModel
    @State private var shareURL: URL?
    @State private var shareError: String?

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

                if decodeModel.snapshot.done {
                    Button {
                        prepareShare()
                    } label: {
                        Label("Share decoded file", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                }

                if let message = decodeModel.statusMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                if let shareError {
                    Text(shareError)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("QRStream")
            .toolbar {
                Button("Reset") {
                    decodeModel.reset()
                    shareURL = nil
                    shareError = nil
                }
            }
            #if canImport(UIKit)
            .sheet(item: $shareURL) { url in
                ActivityView(activityItems: [url])
            }
            #endif
        }
    }

    private func prepareShare() {
        do {
            let data = try decodeModel.resultBytes()
            let ext = sniffExtension(data)
            let filename = "qrstream-\(Int(Date().timeIntervalSince1970)).\(ext)"
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
            try data.write(to: url, options: .atomic)
            shareURL = url
            shareError = nil
        } catch {
            shareError = "Could not prepare file: \(error.localizedDescription)"
        }
    }

    private func sniffExtension(_ data: Data) -> String {
        // Magic-byte sniff covering common types we encode through QRStream.
        guard data.count >= 4 else { return "bin" }
        let bytes = [UInt8](data.prefix(16))

        if bytes.starts(with: [0x25, 0x50, 0x44, 0x46]) { return "pdf" }                         // %PDF
        if bytes.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return "png" }                         // PNG
        if bytes.starts(with: [0xFF, 0xD8, 0xFF]) { return "jpg" }                               // JPEG
        if bytes.starts(with: [0x47, 0x49, 0x46, 0x38]) { return "gif" }                         // GIF8
        if bytes.starts(with: [0x50, 0x4B, 0x03, 0x04]) { return "zip" }                         // PK..
        if bytes.starts(with: [0x1F, 0x8B]) { return "gz" }                                      // gzip
        if bytes.count >= 8 && Array(bytes[4..<8]) == [0x66, 0x74, 0x79, 0x70] { return "mp4" }  // ftyp
        // UTF-8 plain text heuristic: no NUL in the first chunk.
        if !data.prefix(512).contains(0x00),
           String(data: data.prefix(512), encoding: .utf8) != nil {
            return "txt"
        }
        return "bin"
    }
}

#if canImport(UIKit)
extension URL: Identifiable {
    public var id: String { absoluteString }
}

private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif
