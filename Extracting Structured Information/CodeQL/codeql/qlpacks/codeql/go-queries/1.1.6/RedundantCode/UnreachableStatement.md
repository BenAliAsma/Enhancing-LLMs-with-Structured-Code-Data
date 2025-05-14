# Unreachable statement
An unreachable statement often indicates missing code or a latent bug and should be examined carefully.


## Recommendation
Examine the surrounding code to determine why the statement has become unreachable. If it is no longer needed, remove the statement.


## Example
In the following example, the body of the `for` statement cannot terminate normally, so the update statement `i++` becomes unreachable:


```go
package main

func mul(xs []int) int {
	res := 1
	for i := 0; i < len(xs); i++ {
		x := xs[i]
		res *= x
		if res == 0 {
		}
		return 0
	}
	return res
}

```
Most likely, the `return` statement should be moved inside the `if` statement:


```go
package main

func mulGood(xs []int) int {
	res := 1
	for i := 0; i < len(xs); i++ {
		x := xs[i]
		res *= x
		if res == 0 {
			return 0
		}
	}
	return res
}

```

## References
* Wikipedia: [Unreachable code](http://en.wikipedia.org/wiki/Unreachable_code).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
