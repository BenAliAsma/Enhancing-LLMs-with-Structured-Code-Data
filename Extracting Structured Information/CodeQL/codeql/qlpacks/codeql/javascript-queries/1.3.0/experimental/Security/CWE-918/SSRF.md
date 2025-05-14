# Uncontrolled data used in network request
Directly incorporating user input into an HTTP request without validating the input can facilitate server side request forgery attacks, where the attacker essentially controls the request.


## Recommendation
To guard against server side request forgery, it is advisable to avoid putting user input directly into a network request. If using user input is necessary, then is mandatory to validate them. Only allow numeric and alphanumeric values. URL encoding is not a solution in certain scenarios, such as, an architecture build over NGINX proxies.


## Example
The following example shows an HTTP request parameter being used directly in a URL request without validating the input, which facilitates an SSRF attack. The request `axios.get("https://example.com/current_api/"+target)` is vulnerable since attackers can choose the value of `target` to be anything they want. For instance, the attacker can choose `"../super_secret_api"` as the target, causing the URL to become `"https://example.com/super_secret_api"`.

A request to `https://example.com/super_secret_api` may be problematic if that api is not meant to be directly accessible from the attacker's machine.


```javascript
const axios = require('axios');

export const handler = async (req, res, next) => {
  const { target } = req.body;

  try {
    // BAD: `target` is controlled by the attacker
    const response = await axios.get('https://example.com/current_api/' + target);

    // process request response
    use(response);
  } catch (err) {
    // process error
  }
};

```
One way to remedy the problem is to validate the user input to only allow alphanumeric values:


```javascript
const axios = require('axios');
const validator = require('validator');

export const handler = async (req, res, next) => {
  const { target } = req.body;

  if (!validator.isAlphanumeric(target)) {
    return next(new Error('Bad request'));
  }

  try {
    // `target` is validated
    const response = await axios.get('https://example.com/current_api/' + target);

    // process request response
    use(response);
  } catch (err) {
    // process error
  }
};

```

## References
* OWASP: [SSRF](https://www.owasp.org/www-community/attacks/Server_Side_Request_Forgery)
* Common Weakness Enumeration: [CWE-918](https://cwe.mitre.org/data/definitions/918.html).
