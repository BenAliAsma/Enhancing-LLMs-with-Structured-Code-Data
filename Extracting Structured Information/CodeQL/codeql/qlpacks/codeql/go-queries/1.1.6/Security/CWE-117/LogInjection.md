# Log entries created from user input
If unsanitized user input is written to a log entry, a malicious user may be able to forge new log entries.

Forgery can occur if a user provides some input with characters that are interpreted when the log output is displayed. If the log is displayed as a plain text file, then new line characters can be used by a malicious user. If the log is displayed as HTML, then arbitrary HTML may be included to spoof log entries.


## Recommendation
User input should be suitably encoded before it is logged.

If the log entries are plain text then line breaks should be removed from user input, using `strings.Replace` or similar. Care should also be taken that user input is clearly marked in log entries, and that a malicious user cannot cause confusion in other ways.

For log entries that will be displayed in HTML, user input should be HTML encoded using `html.EscapeString` or similar before being logged, to prevent forgery and other forms of HTML injection.


## Example
In the following example, a user name, provided by the user, is logged using a logging framework without any sanitization.


```go
package main

import (
	"log"
	"net/http"
)

// BAD: A user-provided value is written directly to a log.
func handler(req *http.Request) {
	username := req.URL.Query()["username"][0]
	log.Printf("user %s logged in.\n", username)
}

```
In the next example, `strings.Replace` is used to ensure no line endings are present in the user input.


```go
package main

import (
	"log"
	"net/http"
	"strings"
)

// GOOD: The user-provided value is escaped before being written to the log.
func handlerGood(req *http.Request) {
	username := req.URL.Query()["username"][0]
	escapedUsername := strings.ReplaceAll(username, "\n", "")
	escapedUsername = strings.ReplaceAll(escapedUsername, "\r", "")
	log.Printf("user %s logged in.\n", escapedUsername)
}

```

## References
* OWASP: [Log Injection](https://www.owasp.org/index.php/Log_Injection).
* Common Weakness Enumeration: [CWE-117](https://cwe.mitre.org/data/definitions/117.html).
