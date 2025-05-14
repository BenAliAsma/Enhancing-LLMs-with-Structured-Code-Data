# Redundant call to recover
The built-in `recover` function is only useful inside deferred functions. Calling it in a function that is never deferred means that it will always return `nil` and it will never regain control of a panicking goroutine. The same is true of calling `recover` directly in a defer statement.


## Recommendation
Carefully inspect the code to determine whether it is a mistake that should be fixed.


## Example
In the example below, the function `fun1` is intended to recover from the panic. However, the function that is deferred calls another function, which then calls `recover`:


```go
package main

import "fmt"

func callRecover1() {
	if recover() != nil {
		fmt.Printf("recovered")
	}
}

func fun1() {
	defer func() {
		callRecover1()
	}()
	panic("1")
}

```
This problem can be fixed by deferring the call to the function which calls `recover`:


```go
package main

import "fmt"

func callRecover1Good() {
	if recover() != nil {
		fmt.Printf("recovered")
	}
}

func fun1Good() {
	defer callRecover1Good()
	panic("1")
}

```
In the following example, `recover` is called directly in a defer statement, which has no effect, so the panic is not caught.


```go
package main

func fun2() {
	defer recover()
	panic("2")
}

```
We can fix this by instead deferring an anonymous function which calls `recover`.


```go
package main

func fun2Good() {
	defer func() { recover() }()
	panic("2")
}

```

## References
* [Defer, Panic, and Recover - The Go Blog](https://blog.golang.org/defer-panic-and-recover).
