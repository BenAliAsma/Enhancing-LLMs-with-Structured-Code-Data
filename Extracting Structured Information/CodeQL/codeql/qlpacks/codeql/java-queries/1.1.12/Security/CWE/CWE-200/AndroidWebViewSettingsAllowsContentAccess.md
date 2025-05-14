# Android WebView settings allows access to content links
Android can provide access to content providers within a WebView using the `setAllowContentAccess` setting.

Allowing access to content providers via `content://` URLs may allow JavaScript to access protected content.


## Recommendation
If your app does not require access to the `content://` URL functionality, you should explicitly disable the setting by calling `setAllowContentAccess(false)` on the settings of the WebView.


## Example
In the following (bad) example, access to `content://` URLs is explicitly allowed.


```java
WebSettings settings = webview.getSettings();

settings.setAllowContentAccess(true);

```
In the following (good) example, access to `content://` URLs is explicitly denied.


```java
WebSettings settings = webview.getSettings();

settings.setAllowContentAccess(false);

```

## References
* Android Documentation: [setAllowContentAccess](https://developer.android.com/reference/android/webkit/WebSettings#setAllowContentAccess(boolean)).
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
