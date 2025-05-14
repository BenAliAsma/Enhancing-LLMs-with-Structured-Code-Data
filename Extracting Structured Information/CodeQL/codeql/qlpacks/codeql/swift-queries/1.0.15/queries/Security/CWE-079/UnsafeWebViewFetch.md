# Unsafe WebView fetch
Fetching data in a web view without restricting the base URL may allow an attacker to access sensitive local data, for example using `file://`. Data can then be extracted from the software using the URL of a machine under the attacker's control. More generally, an attacker may use a URL under their control as part of a cross-site scripting attack.


## Recommendation
When loading HTML into a web view, always set the `baseURL` to an appropriate URL that you control, or to `about:blank`. Do not use `nil`, as this does not restrict URLs that can be resolved. Also do not use a `baseURL` that could itself be controlled by an attacker.


## Example
In the following example, a call to `UIWebView.loadHTMLString` has the `baseURL` set to `nil`, which does not restrict URLs that can be resolved from within the web page.


```swift

let webview = UIWebView()

...

webview.loadHTMLString(htmlData, baseURL: nil) // BAD

```
To fix the problem, we set the `baseURL` to `about:blank`. This ensures that an attacker cannot resolve URLs that point to the local file system, or to web servers under their control.


```swift

let webview = UIWebView()

...

webview.loadHTMLString(htmlData, baseURL: URL(string: "about:blank")) // GOOD

```

## References
* [iOS Bug Hunting - Web View XSS](https://www.allysonomalley.com/2018/12/03/ios-bug-hunting-web-view-xss/).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-95](https://cwe.mitre.org/data/definitions/95.html).
* Common Weakness Enumeration: [CWE-749](https://cwe.mitre.org/data/definitions/749.html).
