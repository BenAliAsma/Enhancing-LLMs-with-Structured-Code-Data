# Insecure configuration of Helmet security middleware
[Helmet](https://helmetjs.github.io/) is a collection of middleware functions for securing Express apps. It sets various HTTP headers to guard against common web vulnerabilities. This query detects Helmet misconfigurations that can lead to security vulnerabilities, specifically:

* Disabling frame protection
* Disabling Content Security Policy
Content Security Policy (CSP) helps spot and prevent injection attacks such as Cross-Site Scripting (XSS). Removing frame protections exposes an application to attacks such as clickjacking, where an attacker can trick a user into clicking on a button or link on a targeted page when they intended to click on the page carrying out the attack.

Users of the query can extend the set of required Helmet features by adding additional checks for them, using CodeQL [data extensions](https://codeql.github.com/docs/codeql-language-guides/customizing-library-models-for-javascript/) in a [CodeQL model pack](https://docs.github.com/en/code-security/codeql-cli/using-the-advanced-functionality-of-the-codeql-cli/creating-and-working-with-codeql-packs#creating-a-codeql-model-pack). See `CUSTOMIZING.md` in the query source for more information.


## Recommendation
To help mitigate these vulnerabilities, ensure that the following Helmet functions are not disabled, and are configured appropriately to your application:

* `frameguard`
* `contentSecurityPolicy`

## Example
The following code snippet demonstrates Helmet configured in an insecure manner:


```javascript
const helmet = require('helmet');

app.use(helmet({
    frameguard: false,
    contentSecurityPolicy: false
}));
```
In this example, the defaults are used, which enables frame protection and a default Content Security Policy.


```javascript
app.use(helmet());
```
You can also enable a custom Content Security Policy by passing an object to the `contentSecurityPolicy` key. For example, taken from the [Helmet docs](https://helmetjs.github.io/#content-security-policy):


```javascript
app.use(
    helmet({
        contentSecurityPolicy: {
            directives: {
                "script-src": ["'self'", "example.com"],
                "style-src": null,
            },
        },
    })
);
```

## References
* [helmet.js website](https://helmetjs.github.io/)
* [Content Security Policy (CSP) | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy)
* [Mozilla Web Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)
* [Protect against clickjacking | MDN](https://developer.mozilla.org/en-US/docs/Web/Security#protect_against_clickjacking)
* Common Weakness Enumeration: [CWE-693](https://cwe.mitre.org/data/definitions/693.html).
* Common Weakness Enumeration: [CWE-1021](https://cwe.mitre.org/data/definitions/1021.html).
