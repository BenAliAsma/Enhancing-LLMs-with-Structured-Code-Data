# Sensitive cookie without SameSite restrictions
Authentication cookies where the SameSite attribute is set to "None" can potentially be used to perform Cross-Site Request Forgery (CSRF) attacks if no other CSRF protections are in place.

With SameSite set to "None", a third party website may create an authorized cross-site request that includes the cookie. Such a cross-site request can allow that website to perform actions on behalf of a user.


## Recommendation
Set the `SameSite` attribute to `Strict` on all sensitive cookies.


## Example
The following example stores an authentication token in a cookie where the `SameSite` attribute is set to `None`.


```javascript
const http = require('http');

const server = http.createServer((req, res) => {
    res.setHeader("Set-Cookie", `authKey=${makeAuthkey()}; secure; httpOnly; SameSite=None`);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<h2>Hello world</h2>');
});
```
To prevent the cookie from being included in cross-site requests, set the `SameSite` attribute to `Strict`.


```javascript
const http = require('http');

const server = http.createServer((req, res) => {
    res.setHeader("Set-Cookie", `authKey=${makeAuthkey()}; secure; httpOnly; SameSite=Strict`);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<h2>Hello world</h2>');
});
```

## References
* MDN Web Docs: [SameSite cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite).
* OWASP: [SameSite](https://owasp.org/www-community/SameSite).
* Common Weakness Enumeration: [CWE-1275](https://cwe.mitre.org/data/definitions/1275.html).
