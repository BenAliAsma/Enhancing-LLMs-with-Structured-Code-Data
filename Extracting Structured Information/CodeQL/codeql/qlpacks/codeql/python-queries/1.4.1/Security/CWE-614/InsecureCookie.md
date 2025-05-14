# Failure to use secure cookies
Cookies without the `Secure` flag set may be transmitted using HTTP instead of HTTPS, which leaves them vulnerable to reading by a third party.

Cookies without the `HttpOnly` flag set are accessible to JavaScript running in the same origin. In case of a Cross-Site Scripting (XSS) vulnerability, the cookie can be stolen by a malicious script.

Cookies with the `SameSite` attribute set to `'None'` will be sent with cross-origin requests, which can be controlled by third-party JavaScript code and allow for Cross-Site Request Forgery (CSRF) attacks.


## Recommendation
Always set `secure` to `True` or add "; Secure;" to the cookie's raw value.

Always set `httponly` to `True` or add "; HttpOnly;" to the cookie's raw value.

Always set `samesite` to `Lax` or `Strict`, or add "; SameSite=Lax;", or "; Samesite=Strict;" to the cookie's raw header value.


## Example
In the following examples, the cases marked GOOD show secure cookie attributes being set; whereas in the cases marked BAD they are not set.


```python
from flask import Flask, request, make_response, Response


@app.route("/good1")
def good1():
    resp = make_response()
    resp.set_cookie("name", value="value", secure=True, httponly=True, samesite='Strict') # GOOD: Attributes are securely set
    return resp


@app.route("/good2")
def good2():
    resp = make_response()
    resp.headers['Set-Cookie'] = "name=value; Secure; HttpOnly; SameSite=Strict" # GOOD: Attributes are securely set 
    return resp

@app.route("/bad1")
    resp = make_response()
    resp.set_cookie("name", value="value", samesite='None') # BAD: the SameSite attribute is set to 'None' and the 'Secure' and 'HttpOnly' attributes are set to False by default.
    return resp
```

## References
* Detectify: [Cookie lack Secure flag](https://support.detectify.com/support/solutions/articles/48001048982-cookie-lack-secure-flag).
* PortSwigger: [TLS cookie without secure flag set](https://portswigger.net/kb/issues/00500200_tls-cookie-without-secure-flag-set).
* Common Weakness Enumeration: [CWE-614](https://cwe.mitre.org/data/definitions/614.html).
* Common Weakness Enumeration: [CWE-1004](https://cwe.mitre.org/data/definitions/1004.html).
* Common Weakness Enumeration: [CWE-1275](https://cwe.mitre.org/data/definitions/1275.html).
