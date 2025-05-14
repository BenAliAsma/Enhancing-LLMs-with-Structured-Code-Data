# 'HttpOnly' attribute is not set to true
Cookies without `HttpOnly` attribute are accessible to JavaScript running in the same origin. In case of Cross-Site Scripting (XSS) vulnerability the cookie can be stolen by malicious script.


## Recommendation
Protect sensitive cookies, such as related to authentication, by setting `HttpOnly` to `true` to make them not accessible to JavaScript.


## Example
In the following example the default `HttpOnly` value is `false`.


```go
package main

import (
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	c := http.Cookie{
		Name:  "session",
		Value: "secret",
	}
	http.SetCookie(w, &c)
}

func main() {
	http.HandleFunc("/", handler)
}

```
In the example below `HttpOnly` is set to `true`.


```go
package main

import (
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	c := http.Cookie{
		Name:     "session",
		Value:    "secret",
		HttpOnly: true,
	}
	http.SetCookie(w, &c)
}

func main() {
	http.HandleFunc("/", handler)
}

```

## References
* [type Cookie,](https://golang.org/pkg/net/http/#Cookie)
* [Set-Cookie](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie) Header,
* Common Weakness Enumeration: [CWE-1004](https://cwe.mitre.org/data/definitions/1004.html).
