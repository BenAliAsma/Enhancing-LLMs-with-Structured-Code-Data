# Server-side request forgery
Directly incorporating user input in the URL of an outgoing HTTP request can enable a request forgery attack, in which the request is altered to target an unintended API endpoint or resource. If the server performing the request is connected to an internal network, this can give an attacker the means to bypass the network boundary and make requests against internal services. A forged request may perform an unintended action on behalf of the attacker, or cause information leak if redirected to an external server or if the request response is fed back to the user. It may also compromise the server making the request, if the request response is handled in an unsafe way.


## Recommendation
Restrict user inputs in the URL of an outgoing request, in particular:

* Avoid user input in the hostname of the URL. Pick the hostname from an allow-list instead of constructing it directly from user input.
* Take care when user input is part of the pathname of the URL. Restrict the input so that path traversal ("`../`") cannot be used to redirect the request to an unintended endpoint.

## Example
The following example shows an HTTP request parameter being used directly in the URL of a request without validating the input, which facilitates an SSRF attack. The request `http.get(...)` is vulnerable since attackers can choose the value of `target` to be anything they want. For instance, the attacker can choose `"internal.example.com/#"` as the target, causing the URL used in the request to be `"https://internal.example.com/#.example.com/data"`.

A request to `https://internal.example.com` may be problematic if that server is not meant to be directly accessible from the attacker's machine.


```javascript
import http from 'http';

const server = http.createServer(function(req, res) {
    const target = new URL(req.url, "http://example.com").searchParams.get("target");

    // BAD: `target` is controlled by the attacker
    http.get('https://' + target + ".example.com/data/", res => {
        // process request response ...
    });

});

```
One way to remedy the problem is to use the user input to select a known fixed string before performing the request:


```javascript
import http from 'http';

const server = http.createServer(function(req, res) {
    const target = new URL(req.url, "http://example.com").searchParams.get("target");

    let subdomain;
    if (target === 'EU') {
        subdomain = "europe"
    } else {
        subdomain = "world"
    }

    // GOOD: `subdomain` is controlled by the server
    http.get('https://' + subdomain + ".example.com/data/", res => {
        // process request response ...
    });

});

```

## References
* OWASP: [SSRF](https://www.owasp.org/index.php/Server_Side_Request_Forgery)
* Common Weakness Enumeration: [CWE-918](https://cwe.mitre.org/data/definitions/918.html).
