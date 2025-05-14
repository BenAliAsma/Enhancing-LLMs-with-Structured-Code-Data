# Unsafe code constructed from library input
When a library function dynamically constructs code in a potentially unsafe way, then it's important to document to clients of the library that the function should only be used with trusted inputs. If the function is not documented as being potentially unsafe, then a client may incorrectly use inputs containing unsafe code fragments, and thereby leave the client vulnerable to code-injection attacks.


## Recommendation
Properly document library functions that construct code from unsanitized inputs, or avoid constructing code in the first place.


## Example
The following example shows two methods implemented using \`eval\`: a simple deserialization routine and a getter method. If untrusted inputs are used with these methods, then an attacker might be able to execute arbitrary code on the system.


```javascript
export function unsafeDeserialize(value) {
  return eval(`(${value})`);
}

export function unsafeGetter(obj, path) {
    return eval(`obj.${path}`);
}

```
To avoid this problem, either properly document that the function is potentially unsafe, or use an alternative solution such as \`JSON.parse\` or another library, like in the examples below, that does not allow arbitrary code to be executed.


```javascript
export function safeDeserialize(value) {
  return JSON.parse(value);
}

const _ = require("lodash");
export function safeGetter(object, path) {
  return _.get(object, path);
}

```

## References
* OWASP: [Code Injection](https://www.owasp.org/index.php/Code_Injection).
* Wikipedia: [Code Injection](https://en.wikipedia.org/wiki/Code_injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
