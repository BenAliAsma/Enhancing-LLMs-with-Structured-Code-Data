# Failure to abandon session
Reusing a session could allow an attacker to gain unauthorized access to another account. Always ensure that, when a user logs in or out, the current session is abandoned so that a new session may be started.


## Recommendation
Always use `req.session.regenerate(...);` to start a new session when a user logs in or out.


## Example
The following example shows the previous session being used after authentication. This would allow a previous user to use the new user's account.


```javascript
const express = require('express');
const session = require('express-session');
var bodyParser = require('body-parser')
const app = express();
app.use(bodyParser.urlencoded({ extended: false }))
app.use(session({
    secret: 'keyboard cat'
}));

app.post('/login', function (req, res) {
    // Check that username password matches
    if (req.body.username === 'admin' && req.body.password === 'admin') {
        req.session.authenticated = true;
        res.redirect('/');
    } else {
        res.redirect('/login');
    }
});
```
This code example solves the problem by not reusing the session, and instead calling `req.session.regenerate()` to ensure that the session is not reused.


```javascript
const express = require('express');
const session = require('express-session');
var bodyParser = require('body-parser')
const app = express();
app.use(bodyParser.urlencoded({ extended: false }))
app.use(session({
    secret: 'keyboard cat'
}));

app.post('/login', function (req, res) {
    // Check that username password matches
    if (req.body.username === 'admin' && req.body.password === 'admin') {
        req.session.regenerate(function (err) {
            if (err) {
                res.send('Error');
            } else {
                req.session.authenticated = true;
                res.redirect('/');
            }
        });
    } else {
        res.redirect('/login');
    }
});
```

## References
* OWASP: [Session fixation](https://www.owasp.org/index.php/Session_fixation)
* Stack Overflow: [Creating a new session after authentication with Passport](https://stackoverflow.com/questions/22209354/creating-a-new-session-after-authentication-with-passport/30468384#30468384)
* jscrambler.com: [Best practices for secure session management in Node](https://blog.jscrambler.com/best-practices-for-secure-session-management-in-node)
* Common Weakness Enumeration: [CWE-384](https://cwe.mitre.org/data/definitions/384.html).
