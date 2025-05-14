# Insecure http parser
Strict HTTP parsing may cause problems with interoperability with some non-conformant HTTP implementations. But disabling it is strongly discouraged, as it opens the door to several threats including HTTP Request Smuggling.


## Recommendation
Do not enable insecure http parser.


## Example
The following example shows the instantiation of an http server. This server is vulnerable to HTTP Request Smuggling because the `insecureHTTPParser` option of the server instantiation is set to `true`. As a consequence, malformed packets may attempt to exploit any number of weaknesses including ranging from Web Cache Poisoning Attacks to bypassing firewall protection mecahanisms.


```javascript
const http = require('node:http');

http.createServer({
    insecureHTTPParser: true
}, (req, res) => {
    res.end('hello world\n');
});
```
To make sure that packets are parsed correctly, the `invalidHTTPParser` option should have its default value, or be explicitly set to `false`.


## References
* NodeJS: [February 20 Security Release](https://nodejs.org/en/blog/vulnerability/february-2020-security-releases)
* Snyk: [NodeJS Critical HTTP Vulnerability](https://snyk.io/blog/node-js-release-fixes-a-critical-http-security-vulnerability/)
* CWE-444: [HTTP Request/Response Smuggling](https://cwe.mitre.org/data/definitions/444.html)
* Common Weakness Enumeration: [CWE-444](https://cwe.mitre.org/data/definitions/444.html).
