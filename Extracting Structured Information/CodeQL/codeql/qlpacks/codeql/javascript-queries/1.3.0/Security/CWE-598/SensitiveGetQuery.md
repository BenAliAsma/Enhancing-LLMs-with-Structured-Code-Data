# Sensitive data read from GET request
Sensitive information such as user passwords should not be transmitted within the query string of the requested URL. Sensitive information within URLs may be logged in various locations, including the user's browser, the web server, and any forward or reverse proxy servers between the two endpoints. URLs may also be displayed on-screen, bookmarked or emailed around by users. They may be disclosed to third parties via the Referer header when any off-site links are followed. Placing sensitive information into the URL therefore increases the risk that it will be captured by an attacker.


## Recommendation
Use HTTP POST to send sensitive information as part of the request body; for example, as form data.


## Example
The following example shows two route handlers that both receive a username and a password. The first receives this sensitive information from the query parameters of a GET request, which is transmitted in the URL. The second receives this sensitive information from the request body of a POST request.


```javascript
const express = require('express');
const app = express();
app.use(require('body-parser').urlencoded({ extended: false }))

// bad: sensitive information is read from query parameters
app.get('/login1', (req, res) => {
    const user = req.query.user;
    const password = req.query.password;
    if (checkUser(user, password)) {
        res.send('Welcome');
    } else {
        res.send('Access denied');
    }
});

// good: sensitive information is read from post body
app.post('/login2', (req, res) => {
    const user = req.body.user;
    const password = req.body.password;
    if (checkUser(user, password)) {
        res.send('Welcome');
    } else {
        res.send('Access denied');
    }
});

```

## References
* CWE: [CWE-598: Use of GET Request Method with Sensitive Query Strings](https://cwe.mitre.org/data/definitions/598.html)
* PortSwigger (Burp): [Password Submitted using GET Method](https://portswigger.net/kb/issues/00400300_password-submitted-using-get-method)
* OWASP: [Information Exposure through Query Strings in URL](https://owasp.org/www-community/vulnerabilities/Information_exposure_through_query_strings_in_url)
* Common Weakness Enumeration: [CWE-598](https://cwe.mitre.org/data/definitions/598.html).
