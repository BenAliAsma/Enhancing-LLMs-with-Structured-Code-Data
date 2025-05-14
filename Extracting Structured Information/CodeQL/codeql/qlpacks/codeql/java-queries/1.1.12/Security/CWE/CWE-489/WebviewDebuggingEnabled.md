# Android Webview debugging enabled
The `WebView.setWebContentsDebuggingEnabled` method enables or disables the contents of any `WebView` in the application to be debugged.

You should only enable debugging features during development. When you create a production build, you should disable it. If you enable debugging features, this can make your code vulnerable by adding entry points, or leaking sensitive information.


## Recommendation
Ensure that debugging features are not enabled in production builds, such as by guarding calls to `WebView.setWebContentsDebuggingEnabled(true)` by a flag that is only enabled in debug builds.


## Example
In the first (bad) example, WebView debugging is always enabled. whereas the GOOD case only enables it if the `android:debuggable` attribute is set to `true`.


```java
// BAD - debugging is always enabled 
WebView.setWebContentsDebuggingEnabled(true);

// GOOD - debugging is only enabled when this is a debug build, as indicated by the debuggable flag being set.
if (0 != (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE)) {
    WebView.setWebContentsDebuggingEnabled(true);
}
```

## References
* Android Developers: [setWebContentsDebuggingEnabled](https://developer.android.com/reference/android/webkit/WebView.html#setWebContentsDebuggingEnabled(boolean)).
* Android Developers: [Remote debugging WebViews](https://developer.chrome.com/docs/devtools/remote-debugging/webviews/).
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
