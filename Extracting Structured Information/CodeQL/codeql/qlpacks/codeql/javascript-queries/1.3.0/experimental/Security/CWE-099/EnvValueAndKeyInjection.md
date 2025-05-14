# User controlled arbitrary environment variable injection
Controlling the value of arbitrary environment variables from user-controllable data is not safe.


## Recommendation
Restrict this operation only to privileged users or only for some not important environment variables.


## Example
The following example allows unauthorized users to assign a value to any environment variable.


```javascript
const http = require('node:http');

http.createServer((req, res) => {
  const { EnvValue, EnvKey } = req.body;
  process.env[EnvKey] = EnvValue; // NOT OK

  res.end('env has been injected!');
});
```

## References
* [Admin account TakeOver in mintplex-labs/anything-llm](https://huntr.com/bounties/00ec6847-125b-43e9-9658-d3cace1751d6/)
* Common Weakness Enumeration: [CWE-89](https://cwe.mitre.org/data/definitions/89.html).
