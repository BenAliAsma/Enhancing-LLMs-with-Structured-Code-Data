# overly CORS configuration
A server can use `CORS` (Cross-Origin Resource Sharing) to relax the restrictions imposed by the `SOP` (Same-Origin Policy), allowing controlled, secure cross-origin requests when necessary. A server with an overly permissive `CORS` configuration may inadvertently expose sensitive data or lead to `CSRF` which is an attack that allows attackers to trick users into performing unwanted operations in websites they're authenticated to.


## Recommendation
When the `origin` is set to `true`, it signifies that the server is accepting requests from `any` origin, potentially exposing the system to CSRF attacks. This can be fixed using `false` as origin value or using a whitelist.

On the other hand, if the `origin` is set to `null`, it can be exploited by an attacker to deceive a user into making requests from a `null` origin form, often hosted within a sandboxed iframe.

If the `origin` value is user controlled, make sure that the data is properly sanitized.


## Example
In the example below, the `server_1` accepts requests from any origin since the value of `origin` is set to `true`. And `server_2`'s origin is user-controlled.


```javascript
import { ApolloServer } from 'apollo-server';
var https = require('https'),
    url = require('url');

var server = https.createServer(function () { });

server.on('request', function (req, res) {
    // BAD: origin is too permissive
    const server_1 = new ApolloServer({
        cors: { origin: true }
    });

    let user_origin = url.parse(req.url, true).query.origin;
    // BAD: CORS is controlled by user
    const server_2 = new ApolloServer({
        cors: { origin: user_origin }
    });
});
```
In the example below, the `server_1` CORS is restrictive so it's not vulnerable to CSRF attacks. And `server_2`'s is using properly sanitized user-controlled data.


```javascript
import { ApolloServer } from 'apollo-server';
var https = require('https'),
    url = require('url');

var server = https.createServer(function () { });

server.on('request', function (req, res) {
    // GOOD: origin is restrictive
    const server_1 = new ApolloServer({
        cors: { origin: false }
    });

    let user_origin = url.parse(req.url, true).query.origin;
    // GOOD: user data is properly sanitized
    const server_2 = new ApolloServer({
        cors: { origin: (user_origin === "https://allowed1.com" || user_origin === "https://allowed2.com") ? user_origin : false }
    });
});
```

## References
* Mozilla Developer Network: [CORS, Access-Control-Allow-Origin](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin).
* W3C: [CORS for developers, Advice for Resource Owners](https://w3c.github.io/webappsec-cors-for-developers/#resources)
* Common Weakness Enumeration: [CWE-942](https://cwe.mitre.org/data/definitions/942.html).
