# Clear text transmission of sensitive cookie
Cookies that are transmitted in clear text can be intercepted by an attacker. If sensitive cookies are intercepted, the attacker can read the cookie and use it to perform actions on the user's behalf.


## Recommendation
Always transmit sensitive cookies using SSL by setting the `secure` attribute on the cookie.


## Example
The following example stores an authentication token in a cookie that can be transmitted in clear text.


```javascript
const http = require('http');

const server = http.createServer((req, res) => {
    res.setHeader("Set-Cookie", `authKey=${makeAuthkey()}`);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<h2>Hello world</h2>');
});
```
To force the cookie to be transmitted using SSL, set the `secure` attribute on the cookie.


```javascript
const http = require('http');

const server = http.createServer((req, res) => {
    res.setHeader("Set-Cookie", `authKey=${makeAuthkey()}; secure; httpOnly`);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<h2>Hello world</h2>');
});
```

## References
* ExpressJS: [Use cookies securely](https://expressjs.com/en/advanced/best-practice-security.html#use-cookies-securely).
* OWASP: [Set cookie flags appropriately](https://cheatsheetseries.owasp.org/cheatsheets/Nodejs_Security_Cheat_Sheet.html#set-cookie-flags-appropriately).
* Mozilla: [Set-Cookie](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie).
* Common Weakness Enumeration: [CWE-614](https://cwe.mitre.org/data/definitions/614.html).
* Common Weakness Enumeration: [CWE-311](https://cwe.mitre.org/data/definitions/311.html).
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
