# iOS app build & Rust linkage notes

The iOS app (`apps/ios/`) is split between:

- **Swift Package** (`Package.swift`): provides `QRStream` and `QRStreamRust` library targets for unit tests and SwiftPM consumers.
- **Standalone XcodeGen project** (`project.yml`): produces a runnable `QRStreamApp.xcodeproj` for Simulator and device builds. The project is **flat** (no SwiftPM dependency) and links `QRStreamRust.xcframework` directly. A flat target is required because SwiftPM's `qrstreamcoreFFI` C target collides with the modulemap bundled inside the xcframework (both define `module qrstreamcoreFFI`).

## Building the app

### 1. Produce `QRStreamRust.xcframework`

The xcframework is **not committed** (95 MB). Build it from `rust/qrstream-rs/` with iOS targets installed:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release --manifest-path rust/qrstream-rs/Cargo.toml --target aarch64-apple-ios
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release --manifest-path rust/qrstream-rs/Cargo.toml --target aarch64-apple-ios-sim
xcodebuild -create-xcframework \
  -library rust/qrstream-rs/target/aarch64-apple-ios/release/libqrstream_rs.a \
  -headers apps/ios/QRStream/Generated \
  -library rust/qrstream-rs/target/aarch64-apple-ios-sim/release/libqrstream_rs.a \
  -headers apps/ios/QRStream/Generated \
  -output apps/ios/QRStreamRust.xcframework
```

(Re)generate the Swift bindings with:

```bash
cargo build --release --manifest-path rust/qrstream-rs/Cargo.toml
(cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --swift-sources)
(cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --headers)
(cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --modulemap --module-name qrstreamcoreFFI --modulemap-filename module.modulemap)
```

### 2. Generate the Xcode project

The generated `QRStreamApp.xcodeproj` is gitignored. Regenerate it before each build with:

```bash
brew install xcodegen   # one-time
cd apps/ios
DEVELOPMENT_TEAM=<your_team_id> xcodegen generate
```

`DEVELOPMENT_TEAM` is your 10-character Apple Developer Team ID (free Personal Team is fine; find it via Xcode → Settings → Accounts, or in `~/Library/Preferences/com.apple.dt.Xcode.plist` under `IDEProvisioningTeamByIdentifier`). Leave it unset to build for Simulator only.

### 3. Build & run

**Simulator** (no signing needed):

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -project apps/ios/QRStreamApp.xcodeproj -scheme QRStreamApp \
    -destination "platform=iOS Simulator,name=iPhone 17 Pro" \
    CODE_SIGNING_ALLOWED=NO build
```

**Real device** (Apple ID logged into Xcode required for automatic signing):

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -project apps/ios/QRStreamApp.xcodeproj -scheme QRStreamApp \
    -destination "id=<device-udid>" \
    -allowProvisioningUpdates build

# Find the udid via:
xcrun devicectl list devices
# Install + launch:
xcrun devicectl device install app --device <udid> <path-to-built-.app>
xcrun devicectl device process launch --device <udid> dev.qrstream.app
```

After the first install, the device user must visit **Settings → General → VPN & Device Management → Developer App** and trust the developer certificate before the app will launch.

Free Personal Team provisioning profiles expire after **7 days**; rerun the device build to refresh.

## SwiftPM (tests only)

```bash
cd apps/ios && swift test
```

This builds the macOS slice of the xcframework and the Swift Package target tree without going through XcodeGen.
