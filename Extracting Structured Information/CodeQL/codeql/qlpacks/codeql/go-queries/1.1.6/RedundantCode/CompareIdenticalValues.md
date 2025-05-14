# Comparison of identical values
Comparing two identical expressions typically indicates a mistake such as a missing qualifier or a misspelled variable name.


## Recommendation
Carefully inspect the comparison to determine whether it is a symptom of a bug.


## Example
In the example below, the method `Rectangle.contains` is intended to check whether a point `(x, y)` lies inside a rectangle `r` given by its origin `(r.x, r.y)`, its width `r.width`, and its height `r.height`.


```go
package main

type Rectangle struct {
	x, y, width, height float64
}

func (r *Rectangle) containsBad(x, y float64) bool {
	return r.x <= x &&
		y <= y &&
		x <= r.x+r.width &&
		y <= r.y+r.height
}

```
Note, however, that on line 9 the programmer forgot to qualify `r.y`, thus ending up comparing the argument `y` against itself. The comparison should be fixed accordingly:


```go
package main

func (r *Rectangle) containsGood(x, y float64) bool {
	return r.x <= x &&
		r.y <= y &&
		x <= r.x+r.width &&
		y <= r.y+r.height
}

```
