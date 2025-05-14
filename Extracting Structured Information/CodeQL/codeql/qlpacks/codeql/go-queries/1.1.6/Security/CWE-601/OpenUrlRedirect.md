# Open URL redirect
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.

If this is not possible, then the user input should be validated in some other way, for example, by verifying that the target URL is local and does not redirect to a different host.


## Example
The following example shows an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks:


```go
package main

import (
	"net/http"
)

func serve() {
	http.HandleFunc("/redir", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		http.Redirect(w, r, r.Form.Get("target"), 302)
	})
}

```
One way to remedy the problem is to parse the target URL and check that its hostname is empty, which means that it is a relative URL:


```go
package main

import (
	"net/http"
	"net/url"
	"strings"
)

func serve1() {
	http.HandleFunc("/redir", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		targetUrl := r.Form.Get("target")
		// replace all backslashes with forward slashes before parsing the URL
		targetUrl = strings.ReplaceAll(targetUrl, "\\", "/")

		target, err := url.Parse(targetUrl)
		if err != nil {
			// ...
		}

		if target.Hostname() == "" {
			// GOOD: check that it is a local redirect
			http.Redirect(w, r, target.String(), 302)
		} else {
			w.WriteHeader(400)
		}
	})
}

```
Note that some browsers treat backslashes in URLs as forward slashes. To account for this, we replace all backslashes with forward slashes before parsing the URL and checking its hostname.


## References
* OWASP: [ XSS Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).
