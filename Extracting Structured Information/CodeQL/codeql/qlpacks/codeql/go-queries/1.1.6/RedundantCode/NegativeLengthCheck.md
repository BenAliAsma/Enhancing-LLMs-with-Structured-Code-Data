# Redundant check for negative value
The built-in `len` function returns the length of an array, slice or similar, which is never less than zero. Hence, checking whether the result of a call to `len` is negative is either redundant or indicates a logic mistake.

The same applies to the built-in function `cap`, and to unsigned integer values.


## Recommendation
Examine the length check to see whether it is redundant and can be removed, or a mistake that should be fixed.


## Example
The example below shows a function that returns the first element of an array, triggering a panic if the array is empty:


```go
package main

func getFirst(xs []int) int {
	if len(xs) < 0 {
		panic("No elements provided")
	}
	return xs[0]
}

```
However, the emptiness check is ineffective: since `len(xs)` is never less than zero, the condition will never hold and no panic will be triggered. Instead, the index expression `xs[0]` will cause a panic.

The check should be rewritten like this:


```go
package main

func getFirstGood(xs []int) int {
	if len(xs) == 0 {
		panic("No elements provided")
	}
	return xs[0]
}

```

## References
* Package builtin: [func cap](https://golang.org/pkg/builtin/#cap).
* Package builtin: [func len](https://golang.org/pkg/builtin/#len).
