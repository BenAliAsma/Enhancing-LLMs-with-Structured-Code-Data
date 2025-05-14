# Insecure Android WebView Resource Response
Android provides a `WebResourceResponse` class, which allows an Android application to behave as a web server by handling requests of popular protocols such as `http(s)`, `file`, as well as `javascript` and returning a response (including status code, content type, content encoding, headers and the response body). Improper implementation with insufficient input validation can lead to leakage of sensitive configuration files or user data because requests could refer to paths intended to be application-private.


## Recommendation
Unsanitized user-provided URLs must not be used to serve a response directly. When handling a request, always validate that the requested file path is not in the receiver's protected directory. Alternatively the Android class `WebViewAssetLoader` can be used, which safely processes data from resources, assets or a predefined directory.


## Example
The following examples show a bad scenario and two good scenarios respectively. In the bad scenario, a response is served without path validation. In the good scenario, a response is either served with path validation or through the safe `WebViewAssetLoader` implementation.


```java
// BAD: no URI validation
Uri uri = Uri.parse(url);
FileInputStream inputStream = new FileInputStream(uri.getPath());
String mimeType = getMimeTypeFromPath(uri.getPath());
return new WebResourceResponse(mimeType, "UTF-8", inputStream);


// GOOD: check for a trusted prefix, ensuring path traversal is not used to erase that prefix:
// (alternatively use `WebViewAssetsLoader`)
if (uri.getPath().startsWith("/local_cache/") && !uri.getPath().contains("..")) {
    File cacheFile = new File(getCacheDir(), uri.getLastPathSegment());
    FileInputStream inputStream = new FileInputStream(cacheFile);
    String mimeType = getMimeTypeFromPath(uri.getPath());
    return new WebResourceResponse(mimeType, "UTF-8", inputStream);
}

return assetLoader.shouldInterceptRequest(request.getUrl());

```

## References
* Oversecured: [Android: Exploring vulnerabilities in WebResourceResponse](https://blog.oversecured.com/Android-Exploring-vulnerabilities-in-WebResourceResponse/).
* CVE: [CVE-2014-3502: Cordova apps can potentially leak data to other apps via URL loading](https://cordova.apache.org/announcements/2014/08/04/android-351.html).
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
