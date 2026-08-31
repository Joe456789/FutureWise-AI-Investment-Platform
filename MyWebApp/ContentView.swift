import SwiftUI
import WebKit

struct LocalWebView: UIViewRepresentable {
    let fileName: String
    let fileExtension: String

    // 🌟 1. 建立一個 Coordinator 來負責接聽網頁的 alert 視窗
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, WKUIDelegate {
        // 當網頁呼叫 alert() 時，轉換成 iOS 原生的彈出視窗
        func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
            DispatchQueue.main.async {
                if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                   let keyWindow = windowScene.keyWindow,
                   let rootVC = keyWindow.rootViewController {
                    let alert = UIAlertController(title: "系統提示", message: message, preferredStyle: .alert)
                    alert.addAction(UIAlertAction(title: "確定", style: .default, handler: { _ in completionHandler() }))
                    rootVC.present(alert, animated: true, completion: nil)
                } else {
                    completionHandler()
                }
            }
        }
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        configuration.setValue(true, forKey: "allowUniversalAccessFromFileURLs")
        
        let webView = WKWebView(frame: .zero, configuration: configuration)
        
        // 🌟 2. 把負責處理 alert 的小幫手掛載上去
        webView.uiDelegate = context.coordinator
        
        // 🌟 3. 解鎖 Safari 檢閱器 (iOS 16.4+)
        if #available(iOS 16.4, *) {
            webView.isInspectable = true
        }
        
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        if let fileURL = Bundle.main.url(forResource: fileName, withExtension: fileExtension) {
            let readAccessURL = fileURL.deletingLastPathComponent()
            uiView.loadFileURL(fileURL, allowingReadAccessTo: readAccessURL)
        } else {
            print("找不到檔案：\(fileName).\(fileExtension)")
        }
    }
}

struct ContentView: View {
    var body: some View {
        LocalWebView(fileName: "login", fileExtension: "html")
            .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
}
