# Duplicate 'if' branches
If the 'then' and 'else' branches of an 'if' statement are identical, this suggests a copy-paste error where the first branch was copied and then not properly adjusted.


## Recommendation
Examine the two branches to find out what operations were meant to perform. If both the branches and the conditions that they check are identical, then the second branch is duplicate code that can be deleted. If the branches are really meant to perform the same operations, it may be clearer to just have a single branch that checks the disjunction of both conditions.


## Example
The example below shows a buggy implementation of the absolute-value function which checks the sign of its argument, but then returns the same value regardless of the outcome of the check:


```go
package main

func abs(x int) int {
	if x >= 0 {
		return x
	} else {
		return x
	}
}

```
Clearly, the 'else' branch should return `-x` instead:


```go
package main

func absGood(x int) int {
	if x >= 0 {
		return x
	} else {
		return -x
	}
}

```

## References
* The Go Programming Language Specification: [If statements](https://golang.org/ref/spec#If_statements).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
