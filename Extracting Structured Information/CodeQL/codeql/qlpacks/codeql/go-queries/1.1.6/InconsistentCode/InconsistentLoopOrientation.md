# Inconsistent direction of for loop
Most `for` loops either increment a variable until an upper bound is reached, or decrement a variable until a lower bound is reached. If, instead, the variable is incremented but checked against a lower bound, or decremented but checked against an upper bound, then the loop will usually either terminate immediately and never execute its body, or it will keep iterating indefinitely. Neither is likely to be intentional, and is most likely the result of a typo.

The only exception to this are loops whose loop variable is an unsigned integer. In this case, initializing the variable with an upper bound and then decrementing it while checking against the same upper bound is unproblematic: the variable is counted down from the upper bound to zero, and then wraps around to a large positive value, causing the loop to terminate. This is usually intentional, and hence is not flagged by the query.


## Recommendation
Examine the loop carefully to check whether its test expression or update statement are erroneous.


## Example
In the following example, two loops are used to set all elements of a slice `a` outside a range `lower`..`upper` to zero. However, the second loop contains a typo: the loop variable `i` is decremented instead of incremented, so `i` is counted downwards from `upper+1` to `0`, `-1`, `-2` and so on.


```go
package main

func zeroOutExcept(a []int, lower int, upper int) {
	// zero out everything below index `lower`
	for i := lower - 1; i >= 0; i-- {
		a[i] = 0
	}

	// zero out everything above index `upper`
	for i := upper + 1; i < len(a); i-- {
		a[i] = 0
	}
}

```
To correct this issue, change the second loop to increment its loop variable instead:


```go
package main

func zeroOutExcept(a []int, lower int, upper int) {
	// zero out everything below index `lower`
	for i := lower - 1; i >= 0; i-- {
		a[i] = 0
	}

	// zero out everything above index `upper`
	for i := upper + 1; i < len(a); i++ {
		a[i] = 0
	}
}

```

## References
* The Go Programming Language Specification: [For statements](https://golang.org/ref/spec#For_statements).
* Common Weakness Enumeration: [CWE-835](https://cwe.mitre.org/data/definitions/835.html).
