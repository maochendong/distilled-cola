import Cocoa
import Vision

func recognizeText(from imagePath: String) -> String? {
    let url = URL(fileURLWithPath: imagePath)
    guard let image = NSImage(contentsOf: url),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])

    guard let observations = request.results else { return nil }

    let texts = observations
        .sorted { $0.confidence > $1.confidence }
        .compactMap { $0.topCandidates(1).first?.string }

    return texts.isEmpty ? nil : texts.joined(separator: "\n")
}

// Main
let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    let stderrHandle = FileHandle.standardError
    stderrHandle.write("Usage: ocr_tool <image_path>\n".data(using: .utf8)!)
    exit(1)
}

if let result = recognizeText(from: arguments[1]) {
    print(result)
} else {
    exit(1)
}
