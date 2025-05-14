# Hard-coded credentials
Including unencrypted hard-coded authentication credentials in source code is dangerous because the credentials may be easily discovered. For example, the code may be open source, or it may be leaked or accidentally revealed, making the credentials visible to an attacker. This, in turn, might enable them to gain unauthorized access, or to obtain privileged information.


## Recommendation
Remove hard-coded credentials, such as user names, passwords and certificates, from source code. Instead, place them in configuration files, environment variables or other data stores if necessary. If possible, store configuration files including credential data separately from the source code, in a secure location with restricted access.

If the credentials are a placeholder value, make sure the value is obviously a placeholder by using a name such as `"SampleToken"` or `"MyPassword"`.


## Example
The following code example connects to an HTTP request using an hard-codes authentication header:


```javascript
let base64 = require('base-64');

let url = 'http://example.org/auth';
let username = 'user';
let password = 'passwd';

let headers = new Headers();

headers.append('Content-Type', 'text/json');
headers.append('Authorization', 'Basic' + base64.encode(username + ":" + password));

fetch(url, {
          method:'GET',
          headers: headers
       })
.then(response => response.json())
.then(json => console.log(json))
.done();

```
Instead, user name and password can be supplied through the environment variables `username` and `password`, which can be set externally without hard-coding credentials in the source code.


```javascript
let base64 = require('base-64');

let url = 'http://example.org/auth';
let username = process.env.USERNAME;
let password = process.env.PASSWORD;

let headers = new Headers();

headers.append('Content-Type', 'text/json');
headers.append('Authorization', 'Basic' + base64.encode(username + ":" + password));

fetch(url, {
        method:'GET',
        headers: headers
     })
.then(response => response.json())
.then(json => console.log(json))
.done();

```

## Example
The following code example connects to a Postgres database using the `pg` package and hard-codes user name and password:


```javascript
const pg = require("pg");

const client = new pg.Client({
  user: "bob",
  host: "database.server.com",
  database: "mydb",
  password: "correct-horse-battery-staple",
  port: 3211
});
client.connect();

```
Instead, user name and password can be supplied through the environment variables `PGUSER` and `PGPASSWORD`, which can be set externally without hard-coding credentials in the source code.


## References
* OWASP: [Use of hard-coded password](https://www.owasp.org/index.php/Use_of_hard-coded_password).
* Common Weakness Enumeration: [CWE-259](https://cwe.mitre.org/data/definitions/259.html).
* Common Weakness Enumeration: [CWE-321](https://cwe.mitre.org/data/definitions/321.html).
* Common Weakness Enumeration: [CWE-798](https://cwe.mitre.org/data/definitions/798.html).
