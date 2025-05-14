# JWT missing secret or public key verification
Applications decoding JSON Web Tokens (JWT) may be misconfigured due to the `None` algorithm.

The `None` algorithm is selected by calling the `verify()` function with a falsy value instead of a cryptographic secret or key. The `None` algorithm disables the integrity enforcement of a JWT payload and may allow a malicious actor to make unintended changes to a JWT payload leading to critical security issues like privilege escalation.


## Recommendation
Calls to `verify()` functions should use a cryptographic secret or key to decode JWT payloads.


## Example
In the example below, `false` is used to disable the integrity enforcement of a JWT payload. This may allow a malicious actor to make changes to a JWT payload.


```javascript
const jwt = require("jsonwebtoken");

const secret = "my-secret-key";

var token = jwt.sign({ foo: 'bar' }, secret, { algorithm: "none" })
jwt.verify(token, false, { algorithms: ["HS256", "none"] })
```
The following code fixes the problem by using a cryptographic secret or key to decode JWT payloads.


```javascript

const jwt = require("jsonwebtoken");

const secret = "my-secret-key";

var token = jwt.sign({ foo: 'bar' }, secret, { algorithm: "HS256" }) 
jwt.verify(token, secret, { algorithms: ["HS256", "none"] })
```

## References
* Auth0 Blog: [Meet the "None" Algorithm](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/#Meet-the--None--Algorithm).
* Common Weakness Enumeration: [CWE-347](https://cwe.mitre.org/data/definitions/347.html).
