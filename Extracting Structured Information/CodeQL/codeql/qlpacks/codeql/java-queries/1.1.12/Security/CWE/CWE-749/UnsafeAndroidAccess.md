# Unsafe resource fetching in Android WebView
Android WebViews that allow externally controlled URLs to be loaded, and whose JavaScript interface is enabled, are potentially vulnerable to cross-site scripting and sensitive resource disclosure attacks.

A `WebView` whose `WebSettings` object has called `setAllowFileAccessFromFileURLs(true)` or `setAllowUniversalAccessFromFileURLs(true)` must not load any untrusted web content.

Enabling these settings allows malicious scripts loaded in a `file://` context to launch cross-site scripting attacks, accessing arbitrary local files including WebView cookies, session tokens, private app data or even credentials used on arbitrary web sites.

This query detects the following two scenarios:

1. A vulnerability introduced by WebViews when JavaScript is enabled and remote inputs are allowed.
1. A more severe vulnerability when "allow cross-origin resource access" is also enabled. This setting was deprecated in API level 30 (Android 11), but most devices are still affected, especially since some Android phones are updated slowly or no longer updated at all.

## Recommendation
Only allow trusted web content to be displayed in WebViews when JavaScript is enabled. Disallow cross-origin resource access in WebSettings to reduce the attack surface.


## Example
The following example shows both 'BAD' and 'GOOD' configurations. In the 'BAD' configuration, JavaScript and the allow access setting are enabled and URLs are loaded from externally controlled inputs. In the 'GOOD' configuration, JavaScript is disabled or only trusted web content is allowed to be loaded.


```java
public class UnsafeAndroidAccess extends Activity {
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.webview);

		// BAD: Have both JavaScript and cross-origin resource access enabled in webview while
		// taking remote user inputs
		{
			WebView wv = (WebView) findViewById(R.id.my_webview);
			WebSettings webSettings = wv.getSettings();

			webSettings.setJavaScriptEnabled(true);
			webSettings.setAllowUniversalAccessFromFileURLs(true);

			wv.setWebViewClient(new WebViewClient() {
				@Override
				public boolean shouldOverrideUrlLoading(WebView view, String url) {
					view.loadUrl(url);
					return true;
				}
			});

			String thisUrl = getIntent().getExtras().getString("url"); // dangerous remote input from  the intent's Bundle of extras
			wv.loadUrl(thisUrl);
		}

		// BAD: Have both JavaScript and cross-origin resource access enabled in webview while
		// taking remote user inputs
		{
			WebView wv = (WebView) findViewById(R.id.my_webview);
			WebSettings webSettings = wv.getSettings();

			webSettings.setJavaScriptEnabled(true);
			webSettings.setAllowUniversalAccessFromFileURLs(true);

			wv.setWebViewClient(new WebViewClient() {
				@Override
				public boolean shouldOverrideUrlLoading(WebView view, String url) {
					view.loadUrl(url);
					return true;
				}
			});

			String thisUrl = getIntent().getStringExtra("url"); //dangerous remote input from intent extra
			wv.loadUrl(thisUrl);
		}

		// GOOD: Have JavaScript and cross-origin resource access disabled by default on modern Android (Jellybean+) while taking remote user inputs
		{
			WebView wv = (WebView) findViewById(-1);
			WebSettings webSettings = wv.getSettings();

			wv.setWebViewClient(new WebViewClient() {
				@Override
				public boolean shouldOverrideUrlLoading(WebView view, String url) {
					view.loadUrl(url);
					return true;
				}
			});

			String thisUrl = getIntent().getExtras().getString("url"); // remote input
			wv.loadUrl(thisUrl);
		}

		// GOOD: Have JavaScript enabled in webview but remote user input is not allowed
		{
			WebView wv = (WebView) findViewById(-1);
			WebSettings webSettings = wv.getSettings();

			webSettings.setJavaScriptEnabled(true);

			wv.setWebViewClient(new WebViewClient() {
				@Override
				public boolean shouldOverrideUrlLoading(WebView view, String url) {
					view.loadUrl(url);
					return true;
				}
			});

			wv.loadUrl("https://www.mycorp.com");
		}
	}
}
```

## References
* Google Help: [Fixing a File-based XSS Vulnerability](https://support.google.com/faqs/answer/7668153?hl=en)
* OWASP: [Testing JavaScript Execution in WebViews (MSTG-PLATFORM-5)](https://github.com/OWASP/owasp-mstg/blob/master/Document/0x05h-Testing-Platform-Interaction.md#testing-javascript-execution-in-webviews-mstg-platform-5)
* OWASP: [Testing WebView Protocol Handlers (MSTG-PLATFORM-6)](https://github.com/OWASP/owasp-mstg/blob/master/Document/0x05h-Testing-Platform-Interaction.md#testing-webview-protocol-handlers-mstg-platform-6)
* Common Weakness Enumeration: [CWE-749](https://cwe.mitre.org/data/definitions/749.html).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
