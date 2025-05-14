# Useless assignment to local variable
A value is assigned to a variable, but either it is never read, or its value is always overwritten before being read. This means that the original assignment has no effect, and could indicate a logic error or incomplete code.


## Recommendation
Remove assignments to variables that are immediately overwritten, or use the blank identifier `_` as a placeholder for return values that are never used.


## Example
In the following example, a value is assigned to `a`, but then immediately overwritten, a value is assigned to `b` and never used, and finally, the results of a call to `fmt.Println` are assigned to two temporary variables, which are then immediately overwritten by a call to `function`.


```go
package main

import "fmt"

func main() {
	a := calculateValue()
	a = 2

	b := calculateValue()

	ignore, ignore1 := fmt.Println(a)

	ignore, ignore1, err := function()
	if err != nil {
		panic(err)
	}

	fmt.Println(a)
}

```
The result of `calculateValue` is never used, and if `calculateValue` is a side-effect free function, those assignments can be removed. To ignore all the return values of `fmt.Println`, you can simply not assign it to any variables. To ignore only certain return values, use `_`.


```go
package main

import "fmt"

func main() {
	a := 2

	fmt.Println(a)

	_, _, err := function()
	if err != nil {
		panic(err)
	}

	fmt.Println(a)
}

```

## References
* Wikipedia: [Dead store](http://en.wikipedia.org/wiki/Dead_store).
* The Go Programming Language Specification: [Blank identifier](https://golang.org/ref/spec#Blank_identifier).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
