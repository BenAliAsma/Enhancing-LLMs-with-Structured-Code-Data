# Off-by-one comparison against length
Indexing operations on arrays, slices or strings should use an index at most one less than the length. If the index to be accessed is checked for being less than or equal to the length (`<=`), instead of less than the length (`<`), the index could be out of bounds.


## Recommendation
Use less than (`<`) rather than less than or equals (`<=`) when comparing a potential index against a length. For loops that iterate over every element, a better solution is to use a `range` loop instead of looping over explicit indexes.


## Example
The following example shows a method which checks whether a value appears in a comma-separated list of values:


```go
package main

import "strings"

func containsBad(searchName string, names string) bool {
	values := strings.Split(names, ",")
	// BAD: index could be equal to length
	for i := 0; i <= len(values); i++ {
		// When i = length, this access will be out of bounds
		if values[i] == searchName {
			return true
		}
	}
	return false
}

```
A loop using an index variable `i` is used to iterate over the elements in the comma-separated list. However, the terminating condition of the loop is incorrectly specified as `i <= len(values)`. This condition holds when `i` is equal to `len(values)`, but the access `values[i]` in the body of the loop will be out of bounds in this case.

One potential solution would be to replace `i <= len(values)` with `i < len(values)`. A better solution is to use a `range` loop instead, which avoids the need for explicitly manipulating the index variable:


```go
package main

import "strings"

func containsGood(searchName string, names string) bool {
	values := strings.Split(names, ",")
	// GOOD: Avoid using indexes, use range loop instead
	for _, name := range values {
		if name == searchName {
			return true
		}
	}
	return true
}

```

## References
* The Go Programming Language Specification: [For statements](https://golang.org/ref/spec#For_statements).
* The Go Programming Language Specification: [Index expressions](https://golang.org/ref/spec#Index_expressions).
* Common Weakness Enumeration: [CWE-193](https://cwe.mitre.org/data/definitions/193.html).
