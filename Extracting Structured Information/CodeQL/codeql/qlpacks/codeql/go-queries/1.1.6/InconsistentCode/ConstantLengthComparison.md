# Constant length comparison
Indexing operations on arrays, slices, or strings should use an index at most one less than the length. If the operation uses a variable index but checks the length against a constant, this may indicate a logic error which could lead to an out-of-bounds access.


## Recommendation
Inspect the code closely to determine whether the length should be compared to the index variable instead. For loops that iterate over every element, using a `range` loop is better than explicit index manipulation.


## Example
The following example shows a method which checks whether slice `xs` is a prefix of slice `ys`:


```go
package main

func isPrefixOf(xs, ys []int) bool {
	for i := 0; i < len(xs); i++ {
		if len(ys) == 0 || xs[i] != ys[i] {
			return false
		}
	}
	return true
}

```
A loop using an index variable `i` is used to iterate over the elements of `xs` and compare them to the corresponding elements of `ys`. However, the check to ensure that `i` is a valid index into `ys` is incorrectly specified as `len(ys) == 0`. Instead, the check should ensure that `len(ys)` is greater than `i`:


```go
package main

func isPrefixOfGood(xs, ys []int) bool {
	for i := 0; i < len(xs); i++ {
		if len(ys) <= i || xs[i] != ys[i] {
			return false
		}
	}
	return true
}

```

## References
* The Go Programming Language Specification: [For statements](https://golang.org/ref/spec#For_statements).
* The Go Programming Language Specification: [Index expressions](https://golang.org/ref/spec#Index_expressions).
