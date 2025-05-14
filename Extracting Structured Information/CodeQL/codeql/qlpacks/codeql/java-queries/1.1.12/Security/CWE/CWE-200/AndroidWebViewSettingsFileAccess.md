# Android WebSettings file access
Allowing file access in an Android WebView can expose a device's file system to the JavaScript running in that WebView. If the JavaScript contains vulnerabilities or the WebView loads untrusted content, file access allows an attacker to steal the user's data.


## Recommendation
When possible, do not allow file access. The file access settings are disabled by default. You can explicitly disable file access by setting the following settings to `false`:

* `setAllowFileAccess`
* `setAllowFileAccessFromFileURLs`
* `setAllowUniversalAccessFromFileURLs`
If your application requires access to the file system, it is best to avoid using `file://` URLs. Instead, use an alternative that loads files via HTTPS, such as `androidx.webkit.WebViewAssetLoader`.


## Example
In the following (bad) example, the WebView is configured with settings that allow local file access.


```java
WebSettings settings = view.getSettings();

settings.setAllowFileAccess(true);
settings.setAllowFileAccessFromURLs(true);
settings.setAllowUniversalAccessFromURLs(true);

```
In the following (good) example, the WebView is configured to disallow file access.


```java
WebSettings settings = view.getSettings();

settings.setAllowFileAccess(false);
settings.setAllowFileAccessFromURLs(false);
settings.setAllowUniversalAccessFromURLs(false);

```
As mentioned previously, asset loaders can load files without file system access. In the following (good) example, an asset loader is configured to load assets over HTTPS.


```java
WebViewAssetLoader loader = new WebViewAssetLoader.Builder()
    // Replace the domain with a domain you control, or use the default
    // appassets.androidplatform.com
    .setDomain("appassets.example.com")
    .addPathHandler("/resources", new AssetsPathHandler(this))
    .build();

webView.setWebViewClient(new WebViewClientCompat() {
    @Override
    public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
        return assetLoader.shouldInterceptRequest(request.getUrl());
    }
});

webView.loadUrl("https://appassets.example.com/resources/www/index.html");

```

## References
* Android documentation: [WebSettings.setAllowFileAccess](https://developer.android.com/reference/android/webkit/WebSettings#setAllowFileAccess(boolean)).
* Android documentation: [WebSettings.setAllowFileAccessFromFileURLs](https://developer.android.com/reference/android/webkit/WebSettings#setAllowFileAccessFromFileURLs(boolean)).
* Android documentation: [WebSettings.setAllowUniversalAccessFromFileURLs](https://developer.android.com/reference/android/webkit/WebSettings#setAllowUniversalAccessFromFileURLs(boolean)).
* Android documentation: [WebViewAssetLoader](https://developer.android.com/reference/androidx/webkit/WebViewAssetLoader).
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
