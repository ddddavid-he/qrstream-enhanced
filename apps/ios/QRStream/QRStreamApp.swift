import SwiftUI

@main
struct QRStreamApp: App {
    @StateObject private var decodeModel = DecodeSessionModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(decodeModel)
        }
    }
}
