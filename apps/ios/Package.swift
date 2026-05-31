// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QRStreamIOS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "QRStream",
            targets: ["QRStream"]
        ),
        .library(
            name: "QRStreamRust",
            targets: ["QRStreamRust"]
        ),
    ],
    targets: [
        .target(
            name: "QRStream",
            dependencies: ["QRStreamRust"],
            path: "QRStream",
            exclude: ["QRStreamApp.swift", "Generated"]
        ),
        .target(
            name: "QRStreamRust",
            dependencies: ["qrstreamcoreFFI", "QRStreamRustBinary"],
            path: "QRStream/Generated"
        ),
        .target(
            name: "qrstreamcoreFFI",
            path: "QRStreamRustFFI",
            sources: ["src/anchor.c"],
            publicHeadersPath: "include"
        ),
        .binaryTarget(
            name: "QRStreamRustBinary",
            path: "QRStreamRust.xcframework"
        ),
        .testTarget(
            name: "QRStreamTests",
            dependencies: ["QRStream"],
            path: "QRStreamTests"
        ),
    ]
)
