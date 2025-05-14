# Server-side URL redirect
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but which is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.

If this is not possible, then the user input should be validated in some other way, for example, by verifying that the target URL is on the same host as the current page.


## Example
The following example shows an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks:


```javascript
const app = require("express")();

app.get("/redirect", function (req, res) {
  // BAD: a request parameter is incorporated without validation into a URL redirect
  res.redirect(req.query["target"]);
});

```
One way to remedy the problem is to validate the user input against a known fixed string before doing the redirection:


```javascript
const app = require("express")();

const VALID_REDIRECT = "http://cwe.mitre.org/data/definitions/601.html";

app.get("/redirect", function (req, res) {
  // GOOD: the request parameter is validated against a known fixed string
  let target = req.query["target"];
  if (VALID_REDIRECT === target) {
    res.redirect(target);
  } else {
    res.redirect("/");
  }
});

```
Alternatively, we can check that the target URL does not redirect to a different host by parsing it relative to a base URL with a known host and verifying that the host stays the same:


```javascript
const app = require("express")();

function isLocalUrl(path) {
  try {
    return (
      // TODO: consider substituting your own domain for example.com
      new URL(path, "https://example.com").origin === "https://example.com"
    );
  } catch (e) {
    return false;
  }
}

app.get("/redirect", function (req, res) {
  // GOOD: check that we don't redirect to a different host
  let target = req.query["target"];
  if (isLocalUrl(target)) {
    res.redirect(target);
  } else {
    res.redirect("/");
  }
});

```
Note that as written, the above code will allow redirects to URLs on `example.com`, which is harmless but perhaps not intended. You can substitute your own domain (if known) for `example.com` to prevent this.


## References
* OWASP: [ XSS Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).
