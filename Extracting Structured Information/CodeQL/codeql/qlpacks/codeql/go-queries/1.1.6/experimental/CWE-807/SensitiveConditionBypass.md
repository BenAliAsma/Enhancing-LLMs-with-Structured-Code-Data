# User-controlled bypassing of sensitive action
Testing untrusted user input against a fixed constant results in a bypass of the conditional check as the attacker may alter the input to match the constant. When an incorrect check of this type is used to guard a potentially sensitive block, it results an attacker gaining access to the sensitive block.


## Recommendation
Never decide whether to authenticate a user based on data that may be controlled by that user. If necessary, ensure that the data is validated extensively when it is input before any authentication checks are performed.

It is still possible to have a system that "remembers" users, thus not requiring the user to login on every interaction. For example, personalization settings can be applied without authentication because this is not sensitive information. However, users should be allowed to take sensitive actions only when they have been fully authenticated.


## Example
The following example shows a comparison where an user controlled expression is used to guard a sensitive method. This should be avoided.:


```go
package main

import "net/http"

func example(w http.ResponseWriter, r *http.Request) {
	test2 := "test"
	if r.Header.Get("X-Password") != test2 {
		login()
	}
}

```
