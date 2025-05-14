# JavaScript Injection
Evaluating JavaScript that contains a substring from a remote origin may lead to remote code execution. Code written by an attacker can execute unauthorized actions, including exfiltration of local data through a third party web service.


## Recommendation
When loading JavaScript into a web view, evaluate only known, locally-defined source code. If part of the input comes from a remote source, do not inject it into the JavaScript code to be evaluated. Instead, send it to the web view as data using an API such as `WKWebView.callAsyncJavaScript` with the `arguments` dictionary to pass remote data objects.


## Example
In the following (bad) example, a call to `WKWebView.evaluateJavaScript` evaluates JavaScript source code that is tainted with remote data, potentially introducing a code injection vulnerability.


```swift
let webview: WKWebView
let remoteData = try String(contentsOf: URL(string: "http://example.com/evil.json")!)

...

_ = try await webview.evaluateJavaScript("console.log(" + remoteData + ")") // BAD

```
In the following (good) example, we sanitize the remote data by passing it using the `arguments` dictionary of `WKWebView.callAsyncJavaScript`. This ensures that untrusted data cannot be evaluated as JavaScript source code.


```swift
let webview: WKWebView
let remoteData = try String(contentsOf: URL(string: "http://example.com/evil.json")!)

...

_ = try await webview.callAsyncJavaScript(
    "console.log(data)",
    arguments: ["data": remoteData], // GOOD
    contentWorld: .page
)

```

## References
* Apple Developer Documentation: [WKWebView.callAsyncJavaScript(_:arguments:in:contentWorld:)](https://developer.apple.com/documentation/webkit/wkwebview/3824703-callasyncjavascript)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
* Common Weakness Enumeration: [CWE-95](https://cwe.mitre.org/data/definitions/95.html).
* Common Weakness Enumeration: [CWE-749](https://cwe.mitre.org/data/definitions/749.html).
