# JWT missing secret or public key verification
A JSON Web Token (JWT) is used for authenticating and managing users in an application.

Only Decoding JWTs without checking if they have a valid signature or not can lead to security vulnerabilities.


## Recommendation
Don't use methods that only decode JWT, Instead use methods that verify the signature of JWT.


## Example
In the following code, you can see the proper usage of the most popular JWT libraries.


```javascript
const express = require('express')
const app = express()
const jwtJsonwebtoken = require('jsonwebtoken');
const jwt_decode = require('jwt-decode');
const jwt_simple = require('jwt-simple');
const jose = require('jose')
const port = 3000

function getSecret() {
    return "A Safe generated random key"
}

app.get('/jose1', async (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: with signature verification
    await jose.jwtVerify(UserToken, new TextEncoder().encode(getSecret()))
})

app.get('/jose2', async (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: first without signature verification then with signature verification for same UserToken
    jose.decodeJwt(UserToken)
    await jose.jwtVerify(UserToken, new TextEncoder().encode(getSecret()))
})

app.get('/jwtSimple1', (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: first without signature verification then with signature verification for same UserToken
    jwt_simple.decode(UserToken, getSecret(), false);
    jwt_simple.decode(UserToken, getSecret());
})

app.get('/jwtSimple2', (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: with signature verification
    jwt_simple.decode(UserToken, getSecret(), true);
    jwt_simple.decode(UserToken, getSecret());
})

app.get('/jwtJsonwebtoken1', (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: with signature verification
    jwtJsonwebtoken.verify(UserToken, getSecret())
})

app.get('/jwtJsonwebtoken2', (req, res) => {
    const UserToken = req.headers.authorization;
    // GOOD: first without signature verification then with signature verification for same UserToken
    jwtJsonwebtoken.decode(UserToken)
    jwtJsonwebtoken.verify(UserToken, getSecret())
})


app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})
```
In the following code, you can see the improper usage of the most popular JWT libraries.


```javascript
const express = require('express')
const app = express()
const jwtJsonwebtoken = require('jsonwebtoken');
const jwt_decode = require('jwt-decode');
const jwt_simple = require('jwt-simple');
const jose = require('jose')
const port = 3000

function getSecret() {
    return "A Safe generated random key"
}
app.get('/jose', (req, res) => {
    const UserToken = req.headers.authorization;
    // BAD: no signature verification
    jose.decodeJwt(UserToken)
})

app.get('/jwtDecode', (req, res) => {
    const UserToken = req.headers.authorization;
    // BAD: no signature verification
    jwt_decode(UserToken)
})

app.get('/jwtSimple', (req, res) => {
    const UserToken = req.headers.authorization;
    // jwt.decode(token, key, noVerify, algorithm)
    // BAD: no signature verification
    jwt_simple.decode(UserToken, getSecret(), true);
})

app.get('/jwtJsonwebtoken', (req, res) => {
    const UserToken = req.headers.authorization;
    // BAD: no signature verification
    jwtJsonwebtoken.decode(UserToken)
})

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})

```

## References
* [JWT claim has not been verified](https://www.ghostccamm.com/blog/multi_strapi_vulns/#cve-2023-22893-authentication-bypass-for-aws-cognito-login-provider-in-strapi-versions-456)
* Common Weakness Enumeration: [CWE-347](https://cwe.mitre.org/data/definitions/347.html).
