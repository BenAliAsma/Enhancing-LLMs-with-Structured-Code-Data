# Frequency counts for external APIs that are used with untrusted data
Using unsanitized untrusted data in an external API can cause a variety of security issues. This query reports all external APIs that are used with untrusted data, along with how frequently the API is used, and how many unique sources of untrusted data flow to this API. This query is designed primarily to help identify which APIs may be relevant for security analysis of this application.

An external API is defined as a call to a function that is not defined in the source code and is not modeled as a taint step in the default taint library. Calls made in test files are excluded. External APIs may be from the Go standard library, third party dependencies, or from internal dependencies. The query will report the fully-qualified method name, along with either `[param x]`, where `x` indicates the position of the parameter receiving the untrusted data, or `[receiver]` indicating that the untrusted data is used as the receiver of the method call.


## Recommendation
For each result:

* If the result highlights a known sink, no action is required.
* If the result highlights an unknown sink for a problem, then add modeling for the sink to the relevant query.
* If the result represents a call to an external API which transfers taint (for example, from a parameter to its return value), add the appropriate modeling, and re-run the query to determine what new results have appeared due to this additional modeling.
Otherwise, the result is likely uninteresting. Custom versions of this query can extend the `SafeExternalAPIMethod` class to exclude known safe external APIs from future analysis.


## Example

```go
package main

import (
	"fmt"
	"net/http"
)

func serve() {
	http.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		username := r.Form.Get("username")
		if !isValidUsername(username) {
			// BAD: a request parameter is incorporated without validation into the response
			fmt.Fprintf(w, "%q is an unknown user", username)
		} else {
			// TODO: Handle successful login
		}
	})
	http.ListenAndServe(":80", nil)
}

```
If the query were to return the API `fmt.Fprintf [param 2]` then we should first consider whether this a security relevant sink. In this case, this is writing to an HTTP response, so we should consider whether this is an XSS sink. If it is, we should confirm that it is handled by the "Reflected cross-site scripting" query (`go/reflected-xss`).


```go
package main

import (
	"database/sql"
	"fmt"
	"net/http"
)

func handler(db *sql.DB, req *http.Request) {
	q := fmt.Sprintf("SELECT ITEM,PRICE FROM PRODUCT WHERE ITEM_CATEGORY='%s' ORDER BY PRICE",
		req.URL.Query()["category"])
	db.Query(q)
}

```
If the query were to return the API `fmt.Sprintf [param 1]`, then this should be reviewed as a possible taint step, because tainted data would flow from the first argument to the return value of the call.

Note that both examples are correctly handled by the standard taint tracking library and the "Reflected cross-site scripting" query (`go/reflected-xss`).


## References
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
