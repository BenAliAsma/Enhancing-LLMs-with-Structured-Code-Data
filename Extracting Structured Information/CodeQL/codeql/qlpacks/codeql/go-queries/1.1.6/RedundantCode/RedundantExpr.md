# Identical operands
Many arithmetic or logical operators yield a trivial result when applied to identical operands: for instance, `x-x` is zero if `x` is a number, and `NaN` otherwise; `x&x` is always equal to `x`. Code like this is often the result of a typo, such as misspelling a variable name.


## Recommendation
Carefully inspect the expression to ensure it is not a symptom of a bug.


## Example
In the example below, the function `avg` is intended to compute the average of two numbers `x` and `y`. However, the programmer accidentally used `x` twice, so the function just returns `x`:


```go
package main

func avg(x, y float64) float64 {
	return (x + x) / 2
}

```
This problem can be fixed by correcting the typo:


```go
package main

func avgGood(x, y float64) float64 {
	return (x + y) / 2
}

```
