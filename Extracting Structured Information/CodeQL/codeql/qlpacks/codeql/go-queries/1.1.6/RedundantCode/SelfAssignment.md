# Self assignment
Assigning a variable to itself typically indicates a mistake such as a missing qualifier or a misspelled variable name.


## Recommendation
Carefully inspect the assignment to check for misspellings or missing qualifiers.


## Example
In the example below, the struct type `Rect` has two setter methods `setWidth` and `setHeight` that are meant to be used to update the `width` and `height` fields, respectively:


```go
package main

type Rect struct {
	x, y, width, height int
}

func (r *Rect) setWidth(width int) {
	r.width = width
}

func (r *Rect) setHeight(height int) {
	height = height
}

```
Note, however, that in `setHeight` the programmer forgot to qualify the left-hand side of the assignment with the receiver variable `r`, so the method performs a useless assignment of the `width` parameter to itself and leaves the `width` field unchanged.

To fix this issue, insert a qualifier:


```go
package main

func (r *Rect) setHeightGood(height int) {
	r.height = height
}

```

## References
* The Go Programming Language Specification: [Assignments](https://golang.org/ref/spec#Assignments).
* Common Weakness Enumeration: [CWE-480](https://cwe.mitre.org/data/definitions/480.html).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
