# JxBrowser with disabled certificate validation
JxBrowser is a Java library that allows to embed the Chromium browser inside Java applications. Versions smaller than 6.24 by default ignore any HTTPS certificate errors thereby allowing man-in-the-middle attacks.


## Recommendation
Do either of these:

* Update to version 6.24 or 7.x.x as these correctly reject certificate errors by default.
* Add a custom implementation of the `LoadHandler` interface whose `onCertificateError` method always returns **true** indicating that loading should be cancelled. Then use the `setLoadHandler` method with your custom `LoadHandler` on every `Browser` you use.

## Example
The following two examples show two ways of using a `Browser`. In the 'BAD' case, all certificate errors are ignored. In the 'GOOD' case, certificate errors are rejected.


```java
public static void main(String[] args) {
	{
		Browser browser = new Browser();
		browser.loadURL("https://example.com");
		// no further calls
		// BAD: The browser ignores any certificate error by default!
	}

	{
		Browser browser = new Browser();
		browser.setLoadHandler(new LoadHandler() {
			public boolean onLoad(LoadParams params) {
				return true;
			}

			public boolean onCertificateError(CertificateErrorParams params){
				return true; // GOOD: This means that loading will be cancelled on certificate errors
			}
		}); // GOOD: A secure `LoadHandler` is used.
		browser.loadURL("https://example.com");

	}
}
```

## References
* Teamdev: [ Changelog of JxBrowser 6.24](https://jxbrowser-support.teamdev.com/release-notes/2019/v6-24.html).
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
