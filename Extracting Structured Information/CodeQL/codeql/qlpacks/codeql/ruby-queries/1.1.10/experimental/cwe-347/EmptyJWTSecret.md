# JWT encoding using empty key or algorithm
Applications encoding a JSON Web Token (JWT) may be vulnerable when it's not verified or algorithm is `none`.


## Recommendation
JSON Web Tokens should be signed using a strong cryptographic algorithm and non-empty secret.


## Example
In the example below, the secret used is an empty string and none algorithm is used. This may allow a malicious actor to make changes to a JWT payload.


```ruby
require 'jwt'

token1 = JWT.encode({ foo: 'bar' }, "secret", 'none')

token2 = JWT.encode({ foo: 'bar' }, nil, 'HS256')
```
The following code fixes the problem by using a non-empty cryptographic secret or key to encode JWT payloads.


```ruby
require 'jwt'

token = JWT.encode({ foo: 'bar' }, "secret", 'HS256')
```

## References
* Auth0 Blog: [Meet the "None" Algorithm](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/#Meet-the--None--Algorithm).
