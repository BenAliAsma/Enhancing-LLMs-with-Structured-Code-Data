# JWT missing secret or public key verification
Applications decoding a JSON Web Token (JWT) may be vulnerable when the key isn't verified.


## Recommendation
Calls to `verify()` functions should use a cryptographic secret or key to decode JWT payloads.


## Example
In the example below, false is used to disable the integrity enforcement of a JWT payload and none algorithm is used. This may allow a malicious actor to make changes to a JWT payload.


```ruby
require 'jwt'

token = JWT.encode({ foo: 'bar' }, nil, 'none')

decoded1 = JWT.decode(token, nil, false, algorithm: 'HS256')

decoded2 = JWT.decode(token, "secret", true, algorithm: 'none')
```
The following code fixes the problem by using a cryptographic secret or key to decode JWT payloads.


```ruby
require 'jwt'

token = JWT.encode({ foo: 'bar' }, nil, 'HS256')

decoded = JWT.decode(token, "secret", true, algorithm: 'HS256')

```

## References
* Auth0 Blog: [Meet the "None" Algorithm](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/#Meet-the--None--Algorithm).
* Common Weakness Enumeration: [CWE-347](https://cwe.mitre.org/data/definitions/347.html).
