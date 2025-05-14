# Incomplete regular expression for hostnames
Sanitizing untrusted URLs is an important technique for preventing attacks such as request forgeries and malicious redirections. Often, this is done by checking that the host of a URL is in a set of allowed hosts.

If a regular expression implements such a check, it is easy to accidentally make the check too permissive by not escaping regular-expression meta-characters such as `.`.

Even if the check is not used in a security-critical context, the incomplete check may still cause undesirable behavior when it accidentally succeeds.


## Recommendation
Escape all meta-characters appropriately when constructing regular expressions for security checks, paying special attention to the `.` meta-character.


## Example
The following example code checks that a URL redirection will reach the `example.com` domain, or one of its subdomains.


```go
package main

import (
	"errors"
	"net/http"
	"regexp"
)

func checkRedirect(req *http.Request, via []*http.Request) error {
	// BAD: the host of `req.URL` may be controlled by an attacker
	re := "^((www|beta).)?example.com/"
	if matched, _ := regexp.MatchString(re, req.URL.Host); matched {
		return nil
	}
	return errors.New("Invalid redirect")
}

```
The check is however easy to bypass because the unescaped `.` allows for any character before `example.com`, effectively allowing the redirect to go to an attacker-controlled domain such as `wwwXexample.com`.

Address this vulnerability by escaping `.` appropriately:


```go
package main

import (
	"errors"
	"net/http"
	"regexp"
)

func checkRedirectGood(req *http.Request, via []*http.Request) error {
	// GOOD: the host of `req.URL` must be `example.com`, `www.example.com` or `beta.example.com`
	re := "^((www|beta)\\.)?example\\.com/"
	if matched, _ := regexp.MatchString(re, req.URL.Host); matched {
		return nil
	}
	return errors.New("Invalid redirect")
}

```
You may also want to consider using raw string literals to avoid having to escape backslashes:


```go
package main

import (
	"errors"
	"net/http"
	"regexp"
)

func checkRedirectGood2(req *http.Request, via []*http.Request) error {
	// GOOD: the host of `req.URL` must be `example.com`, `www.example.com` or `beta.example.com`
	re := `^((www|beta)\.)?example\.com/`
	if matched, _ := regexp.MatchString(re, req.URL.Host); matched {
		return nil
	}
	return errors.New("Invalid redirect")
}

```

## References
* OWASP: [SSRF](https://www.owasp.org/index.php/Server_Side_Request_Forgery)
* OWASP: [Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
