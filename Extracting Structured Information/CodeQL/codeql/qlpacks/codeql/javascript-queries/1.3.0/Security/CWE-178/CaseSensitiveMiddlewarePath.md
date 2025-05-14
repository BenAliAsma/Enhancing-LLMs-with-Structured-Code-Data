# Case-sensitive middleware path
Using a case-sensitive regular expression path in a middleware route enables an attacker to bypass that middleware when accessing an endpoint with a case-insensitive path. Paths specified using a string are case-insensitive, whereas regular expressions are case-sensitive by default.


## Recommendation
When using a regular expression as a middleware path, make sure the regular expression is case-insensitive by adding the `i` flag.


## Example
The following example restricts access to paths in the `/admin` path to users logged in as administrators:


```javascript
const app = require('express')();

app.use(/\/admin\/.*/, (req, res, next) => {
    if (!req.user.isAdmin) {
        res.status(401).send('Unauthorized');
    } else {
        next();
    }
});

app.get('/admin/users/:id', (req, res) => {
    res.send(app.database.users[req.params.id]);
});

```
A path such as `/admin/users/45` can only be accessed by an administrator. However, the path `/ADMIN/USERS/45` can be accessed by anyone because the upper-case path doesn't match the case-sensitive regular expression, whereas Express considers it to match the path string `/admin/users`.

The issue can be fixed by adding the `i` flag to the regular expression:


```javascript
const app = require('express')();

app.use(/\/admin\/.*/i, (req, res, next) => {
    if (!req.user.isAdmin) {
        res.status(401).send('Unauthorized');
    } else {
        next();
    }
});

app.get('/admin/users/:id', (req, res) => {
    res.send(app.database.users[req.params.id]);
});

```

## References
* MDN [Regular Expression Flags](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions#advanced_searching_with_flags).
* Common Weakness Enumeration: [CWE-178](https://cwe.mitre.org/data/definitions/178.html).
