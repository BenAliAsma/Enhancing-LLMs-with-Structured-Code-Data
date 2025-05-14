# Useless assignment to field
A value is assigned to a field, but its value is never read. This means that the assignment has no effect, and could indicate a logic error or incomplete code.


## Recommendation
Examine the assignment closely to determine whether it is redundant, or whether it is perhaps a symptom of another bug.


## Example
The following example shows a simple `struct` type wrapping an integer counter with a method `reset` that sets the counter to zero.


```go
package main

type counter struct {
	val int
}

func (c counter) reset() {
	c.val = 0
}

```
However, the receiver variable of `reset` is declared to be of type `counter`, not `*counter`, so the receiver value is passed into the method by value, not by reference. Consequently, the method does not actually mutate its receiver as intended.

To fix this, change the type of the receiver variable to `*counter`:


```go
package main

func (c *counter) resetGood() {
	c.val = 0
}

```

## References
* Go Frequently Asked Questions: [Should I define methods on values or pointers?](https://golang.org/doc/faq#methods_on_values_or_pointers)
* The Go Programming Language Specification: [Method declarations](https://golang.org/ref/spec#Method_declarations).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
