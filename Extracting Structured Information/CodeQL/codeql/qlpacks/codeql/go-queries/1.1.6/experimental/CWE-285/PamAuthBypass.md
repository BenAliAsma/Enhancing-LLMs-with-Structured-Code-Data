# PAM authorization bypass due to incorrect usage
Using only a call to `pam.Authenticate` to check the validity of a login can lead to authorization bypass vulnerabilities.

A `pam.Authenticate` call only verifies the credentials of a user. It does not check if a user has an appropriate authorization to actually login. This means a user with an expired login or a password can still access the system.


## Recommendation
A call to `pam.Authenticate` should be followed by a call to `pam.AcctMgmt` to check if a user is allowed to login.


## Example
In the following example, the code only checks the credentials of a user. Hence, in this case, a user with expired credentials can still login. This can be verified by creating a new user account, expiring it with ``` chage -E0 `username`  ``` and then trying to log in.


```go
package main

import "fmt"

func bad() (string, error) {
	// ...
	t, err := pam.StartFunc("", "username", func(s pam.Style, msg string) (string, error) {
		switch s {
		case pam.PromptEchoOff:
			return string(pass), nil
		}
		return "", fmt.Errorf("unsupported message style")
	})
	if err != nil {
		return nil, err
	}

	if err := t.Authenticate(0); err != nil {
		return nil, fmt.Errorf("Authenticate: %w", err)
	}
}

```
This can be avoided by calling `pam.AcctMgmt` call to verify access as has been done in the snippet shown below.


```go
package main

import "fmt"

func good() (string, error) {
	t, err := pam.StartFunc("", "username", func(s pam.Style, msg string) (string, error) {
		switch s {
		case pam.PromptEchoOff:
			return string(pass), nil
		}
		return "", fmt.Errorf("unsupported message style")
	})
	if err != nil {
		return nil, err
	}

	if err := t.Authenticate(0); err != nil {
		return nil, fmt.Errorf("Authenticate: %w", err)
	}
	if err := t.AcctMgmt(0); err != nil {
		return nil, fmt.Errorf("AcctMgmt: %w", err)
	}
}

```

## References
* Man-Page: [pam_acct_mgmt](https://man7.org/linux/man-pages/man3/pam_acct_mgmt.3.html)
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
* Common Weakness Enumeration: [CWE-285](https://cwe.mitre.org/data/definitions/285.html).
