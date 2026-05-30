# QRStreamRust SwiftPM linkage notes

This directory contains generated UniFFI Swift bindings under `QRStream/Generated/`, but the SwiftPM target intentionally excludes them until a Rust static library or XCFramework is available.

Current status:

- Rust UniFFI scaffolding builds on the host target with:
  ```bash
  cargo build --manifest-path rust/qrstream-rs/Cargo.toml
  ```
- PyO3 compatibility still works with:
  ```bash
  uvx maturin develop --manifest-path rust/qrstream-rs/Cargo.toml
  ```
- Swift bindings were generated from the host staticlib with:
  ```bash
  cargo build --release --manifest-path rust/qrstream-rs/Cargo.toml
  (cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --swift-sources)
  (cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --headers)
  (cd rust/qrstream-rs && cargo run --bin uniffi-bindgen-swift -- target/release/libqrstream_rs.a ../../apps/ios/QRStream/Generated --modulemap --module-name qrstream_coreFFI --modulemap-filename module.modulemap)
  ```

To fully link the generated Swift into the app, install Rust iOS targets and build an XCFramework:

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

The current environment uses Homebrew Rust without `rustup`, so the iOS Rust std targets are unavailable locally:

```text
can't find crate for `core`
the `aarch64-apple-ios-sim` target may not be installed
```

Until the XCFramework exists, `Package.swift` excludes `QRStream/Generated` and `RustDecodeSession.swift` is guarded by `#if canImport(qrstreamcoreFFI)`.
