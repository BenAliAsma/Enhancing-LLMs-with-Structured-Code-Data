# Slice memory allocation with excessive size value
Using untrusted input to allocate slices with the built-in `make` function could lead to excessive memory allocation and potentially cause the program to crash due to running out of memory. This vulnerability could be exploited to perform a denial-of-service attack by consuming all available server resources.


## Recommendation
Implement a maximum allowed value for size allocations with the built-in `make` function to prevent excessively large allocations.


## Example
In the following example snippet, the `n` parameter is user-controlled.

If the external user provides an excessively large value, the application allocates a slice of size `n` without further verification, potentially exhausting all the available memory.


```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

func OutOfMemoryBad(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()

	queryStr := query.Get("n")
	collectionSize, err := strconv.Atoi(queryStr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := make([]string, collectionSize)
	for i := 0; i < collectionSize; i++ {
		result[i] = fmt.Sprintf("Item %d", i+1)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

```
One way to prevent this vulnerability is by implementing a maximum allowed value for the user-controlled input, as seen in the following example:


```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

func OutOfMemoryGood(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	MaxValue := 6
	queryStr := query.Get("n")
	collectionSize, err := strconv.Atoi(queryStr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if collectionSize < 0 || collectionSize > MaxValue {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
	result := make([]string, collectionSize)
	for i := 0; i < collectionSize; i++ {
		result[i] = fmt.Sprintf("Item %d", i+1)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

```

## References
* OWASP: [Denial of Service Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html)
* Common Weakness Enumeration: [CWE-770](https://cwe.mitre.org/data/definitions/770.html).
