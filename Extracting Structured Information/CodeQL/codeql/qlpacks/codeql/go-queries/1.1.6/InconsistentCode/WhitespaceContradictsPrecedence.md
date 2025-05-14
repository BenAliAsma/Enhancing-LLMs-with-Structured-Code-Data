# Whitespace contradicts operator precedence
Nested expressions where the spacing around operators suggests a different grouping than that imposed by the Go operator precedence rules are problematic: they could indicate a bug where the author of the code misunderstood the precedence rules. Even if there is no a bug, the spacing could be confusing to people who read the code.


## Recommendation
Make sure that the spacing around operators reflects operator precedence, or use parentheses to clarify grouping.


## Example
Consider the following function intended for checking whether the bit at position \`pos\` of the variable \`x\` is set:


```go
package main

func isBitSetBad(x int, pos uint) bool {
	return x&1<<pos != 0
}

```
Here, the spacing around `&` and `<<` suggests the grouping `x & (1<<pos)`. However, in Go `&` and `<<` have the same precedence and hence are evaluated left to right, so the expression is actually equivalent to `(x & 1) << pos`.

To fix this issue and give the expression its intended semantics, parentheses should be used like this:


```go
package main

func isBitSetGood(x int, pos uint) bool {
	return x&(1<<pos) != 0
}

```

## References
* The Go Programming Language Specification: [Operator precedence](https://golang.org/ref/spec#Operator_precedence).
* Common Weakness Enumeration: [CWE-783](https://cwe.mitre.org/data/definitions/783.html).
