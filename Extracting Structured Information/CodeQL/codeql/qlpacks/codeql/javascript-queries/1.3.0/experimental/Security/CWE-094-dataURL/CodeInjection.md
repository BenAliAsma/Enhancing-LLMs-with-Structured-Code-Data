# Code injection from dynamically imported code
Directly evaluating user input (for example, an HTTP request parameter) as code without properly sanitizing the input first allows an attacker arbitrary code execution. This can occur when user input is treated as JavaScript, or passed to a framework which interprets it as an expression to be evaluated. Examples include AngularJS expressions or JQuery selectors.


## Recommendation
Avoid including user input in any expression which may be dynamically evaluated. If user input must be included, use context-specific escaping before including it. It is important that the correct escaping is used for the type of evaluation that will occur.


## Example
The following example shows part of the page URL being evaluated as JavaScript code on the server. This allows an attacker to provide JavaScript within the URL and send it to server. client side attacks need victim users interaction like clicking on a attacker provided URL.


```javascript
const { Worker } = require('node:worker_threads');
var app = require('express')();

app.post('/path', async function (req, res) {
    const payload = req.query.queryParameter // like:  payload = 'data:text/javascript,console.log("hello!");//'
    const payloadURL = new URL(payload)
    new Worker(payloadURL);
});

app.post('/path2', async function (req, res) {
    const payload = req.query.queryParameter // like:  payload = 'data:text/javascript,console.log("hello!");//'
    await import(payload)
});


```

## References
* OWASP: [Code Injection](https://www.owasp.org/index.php/Code_Injection).
* Wikipedia: [Code Injection](https://en.wikipedia.org/wiki/Code_injection).
* PortSwigger Research Blog: [Server-Side Template Injection](https://portswigger.net/research/server-side-template-injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
* Common Weakness Enumeration: [CWE-95](https://cwe.mitre.org/data/definitions/95.html).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
