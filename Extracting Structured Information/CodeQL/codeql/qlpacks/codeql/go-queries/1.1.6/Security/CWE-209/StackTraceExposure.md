# Information exposure through a stack trace
Software developers often add stack traces to error messages, as a debugging aid. Whenever that error message occurs for an end user, the developer can use the stack trace to help identify how to fix the problem. In particular, stack traces can tell the developer more about the sequence of events that led to a failure, as opposed to merely the final state of the software when the error occurred.

Unfortunately, the same information can be useful to an attacker. The sequence of class names in a stack trace can reveal the structure of the application as well as any internal components it relies on.


## Recommendation
Send the user a more generic error message that reveals less information. Either suppress the stack trace entirely, or log it only on the server.


## Example
In the following example, a panic is handled in two different ways. In the first version, labeled BAD, a detailed stack trace is written to a user-facing HTTP response object, which may disclose sensitive information. In the second version, the error message is logged only on the server. That way, the developers can still access and use the error log, but remote users will not see the information.


```go
package example

import (
	"log"
	"net/http"
	"runtime"
)

func handlePanic(w http.ResponseWriter, r *http.Request) {
	buf := make([]byte, 2<<16)
	buf = buf[:runtime.Stack(buf, true)]
	// BAD: printing a stack trace back to the response
	w.Write(buf)
	// GOOD: logging the response to the server and sending
	// a more generic message.
	log.Printf("Panic: %s", buf)
	w.Write([]byte("An unexpected runtime error occurred"))
}

```

## References
* OWASP: [Improper Error Handling](https://owasp.org/www-community/Improper_Error_Handling).
* Common Weakness Enumeration: [CWE-209](https://cwe.mitre.org/data/definitions/209.html).
* Common Weakness Enumeration: [CWE-497](https://cwe.mitre.org/data/definitions/497.html).
