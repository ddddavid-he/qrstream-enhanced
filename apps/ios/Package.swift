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
    ],
    targets: [
        .target(
            name: "QRStream",
            path: "QRStream",
            exclude: ["QRStreamApp.swift"]
        ),
        .testTarget(
            name: "QRStreamTests",
            dependencies: ["QRStream"],
            path: "QRStreamTests"
        ),
    ]
)
