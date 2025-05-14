# Timing attacks due to comparison of sensitive secrets
Using a non-constant time comparision to compare sensitive information can lead to auth vulnerabilities.


## Recommendation
Use of a constant time comparision function such as `crypto/subtle` package's ` ConstantTimeCompare` function can prevent this vulnerability.


## Example
In the following examples, the code accepts a secret via a HTTP header in variable ` secretHeader` and a secret from the user in the `headerSecret` variable, which are then compared with a system stored secret to perform authentication.


```go
package main

import (
	"fmt"
	"net/http"
)

func bad(w http.ResponseWriter, req *http.Request, secret []byte) (interface{}, error) {

	secretHeader := "X-Secret"

	headerSecret := req.Header.Get(secretHeader)
	secretStr := string(secret)
	if len(secret) != 0 && headerSecret != secretStr {
		return nil, fmt.Errorf("header %s=%s did not match expected secret", secretHeader, headerSecret)
	}
	return nil, nil
}

```
In the following example, the input provided by the user is compared using the ` ConstantTimeComapre` function. This ensures that timing attacks are not possible in this case.


```go
package main

import (
	"crypto/subtle"
	"fmt"
	"net/http"
)

func good(w http.ResponseWriter, req *http.Request, secret []byte) (interface{}, error) {

	secretHeader := "X-Secret"

	headerSecret := req.Header.Get(secretHeader)
	if len(secret) != 0 && subtle.ConstantTimeCompare(secret, []byte(headerSecret)) != 1 {
		return nil, fmt.Errorf("header %s=%s did not match expected secret", secretHeader, headerSecret)
	}
	return nil, nil
}

```

## References
* National Vulnerability Database: [ CVE-2022-24912](https://nvd.nist.gov/vuln/detail/CVE-2022-24912).
* Verbose Logging:[ A timing attack in action ](https://verboselogging.com/2012/08/20/a-timing-attack-in-action)
* Common Weakness Enumeration: [CWE-203](https://cwe.mitre.org/data/definitions/203.html).
