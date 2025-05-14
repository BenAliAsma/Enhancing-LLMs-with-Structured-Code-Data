# Android WebView JavaScript settings
Enabling JavaScript in an Android WebView allows the execution of JavaScript code in the context of the running application. This creates a cross-site scripting vulnerability.

For example, if your application's WebView allows for visiting web pages that you do not trust, it is possible for an attacker to lead the user to a page which loads malicious JavaScript.

You can enable or disable Javascript execution using the `setJavaScriptEnabled` method of the settings of a WebView.


## Recommendation
JavaScript execution is disabled by default. You can explicitly disable it by calling `setJavaScriptEnabled(false)` on the settings of the WebView.

If JavaScript is necessary, only load content from trusted servers using encrypted channels, such as HTTPS with certificate verification.


## Example
In the following (bad) example, a WebView has JavaScript enabled in its settings:


```java
WebSettings settings = webview.getSettings();
settings.setJavaScriptEnabled(true);

```
In the following (good) example, a WebView explicitly disallows JavaScript execution:


```java
WebSettings settings = webview.getSettings();
settings.setJavaScriptEnabled(false);

```

## References
* Android documentation: [setJavaScriptEnabled](https://developer.android.com/reference/android/webkit/WebSettings#setJavaScriptEnabled(boolean))
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
