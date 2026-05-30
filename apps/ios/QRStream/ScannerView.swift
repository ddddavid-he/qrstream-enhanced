import SwiftUI

#if canImport(UIKit) && canImport(AVFoundation)
import AVFoundation
import UIKit

public struct ScannerView: UIViewControllerRepresentable {
    public let onQRCode: (String) -> Void

    public init(onQRCode: @escaping (String) -> Void) {
        self.onQRCode = onQRCode
    }

    public func makeUIViewController(context: Context) -> ScannerViewController {
        let controller = ScannerViewController()
        controller.onQRCode = onQRCode
        return controller
    }

    public func updateUIViewController(_ uiViewController: ScannerViewController, context: Context) {}
}

public final class ScannerViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    public var onQRCode: ((String) -> Void)?

    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var lastPayload: String?

    public override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureSession()
    }

    public override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    public override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [session] in
                session.startRunning()
            }
        }
    }

    public override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if session.isRunning {
            session.stopRunning()
        }
    }

    private func configureSession() {
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else {
            showUnavailableLabel()
            return
        }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        session.addInput(input)

        let metadataOutput = AVCaptureMetadataOutput()
        guard session.canAddOutput(metadataOutput) else {
            session.commitConfiguration()
            showUnavailableLabel()
            return
        }
        session.addOutput(metadataOutput)
        metadataOutput.setMetadataObjectsDelegate(self, queue: .main)
        if metadataOutput.availableMetadataObjectTypes.contains(.qr) {
            metadataOutput.metadataObjectTypes = [.qr]
        }
        session.commitConfiguration()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        previewLayer = layer
    }

    private func showUnavailableLabel() {
        let label = UILabel()
        label.text = "Camera unavailable"
        label.textColor = .white
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
    }

    public func metadataOutput(
        _ output: AVCaptureMetadataOutput,
        didOutput metadataObjects: [AVMetadataObject],
        from connection: AVCaptureConnection
    ) {
        guard let object = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
              object.type == .qr,
              let payload = object.stringValue,
              payload != lastPayload
        else {
            return
        }
        lastPayload = payload
        onQRCode?(payload)
    }
}
#else
public struct ScannerView: View {
    public let onQRCode: (String) -> Void

    public init(onQRCode: @escaping (String) -> Void) {
        self.onQRCode = onQRCode
    }

    public var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "qrcode.viewfinder")
                .font(.system(size: 48))
            Text("QR scanner preview is available on iOS devices.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Simulate QR Frame") {
                onQRCode("placeholder")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.opacity(0.08))
    }
}
#endif
