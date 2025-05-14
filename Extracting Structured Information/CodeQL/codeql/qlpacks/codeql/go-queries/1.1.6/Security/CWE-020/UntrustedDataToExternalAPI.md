# Untrusted data passed to external API
Using unsanitized untrusted data in an external API can cause a variety of security issues. This query reports external APIs that use untrusted data. The results have very little filtering so that you can audit almost all examples. The query provides data for security reviews of the application. It can also be used to identify external APIs that should be modeled as either taint steps, or sinks for specific problems.

An external API is defined as a call to a function that is not defined in the source code and is not modeled as a taint step in the default taint library. Calls made in test files are excluded. External APIs may be from the Go standard library, third-party dependencies, or from internal dependencies. The query reports uses of untrusted data in either the receiver or as one of the arguments of external APIs.


## Recommendation
For each result:

* If the result highlights a known sink, confirm that the result is reported by the relevant query, or that the result is a false positive because this data is sanitized.
* If the result highlights an unknown sink for a problem, then add modeling for the sink to the relevant query, and confirm that the result is either found, or is safe due to appropriate sanitization.
* If the result represents a call to an external API that transfers taint (for example, from a parameter to its return value), add the appropriate modeling, and re-run the query to determine what new results have appeared due to this additional modeling.
Otherwise, the result is likely uninteresting. Custom versions of this query can extend the `SafeExternalAPIMethod` class to exclude known safe external APIs from future analysis.


## Example
In this first example, a request parameter is read from `http.Request` and then ultimately used in a call to the `fmt.Fprintf` external API:


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
This is an XSS sink. The "Reflected cross-site scripting" query (`go/reflected-xss`) should therefore be reviewed to confirm that this sink is appropriately modeled, and if it is, to confirm that the query reports this particular result, or that the result is a false positive due to some existing sanitization.

In this second example, again a request parameter is read from `http.Request`.


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
If the query reported the call to `fmt.Sprintf`, this would suggest that this external API is not currently modeled as a taint step in the taint tracking library. The next step would be to model this as a taint step, then re-run the query to determine what additional results might be found. In this example, an SQL injection vulnerability would be reported.

Note that both examples are correctly handled by the standard taint tracking library, the "Reflected cross-site scripting" query (`go/reflected-xss`), and the "Database query built from user-controlled sources" query (`go/sql-injection`).


## References
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
