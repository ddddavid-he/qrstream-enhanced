import SwiftUI

@main
struct QRStreamApp: App {
    @StateObject private var decodeModel = DecodeSessionModel(session: RustDecodeSession())

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(decodeModel)
        }
    }
}
