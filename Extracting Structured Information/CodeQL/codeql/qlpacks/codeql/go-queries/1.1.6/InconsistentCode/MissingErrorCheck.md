# Missing error check
When a function call returns two values, a pointer and a (subtype of) error, it is conventional to assume that the pointer might be nil until either the pointer or error value has been checked.

If the pointer is dereferenced without a check, an unexpected nil pointer dereference panic may occur.


## Recommendation
Ensure that the returned pointer is either directly checked against nil, or the error value is checked before using the returned pointer.


## Example
In the example below, `user` dereferences `ptr` without checking either `ptr` or `err`. This might lead to a panic.


```go
package main

import (
	"fmt"
	"os"
)

func user(input string) {

	ptr, err := os.Open(input)
	// BAD: ptr is dereferenced before either it or `err` has been checked.
	fmt.Printf("Opened %v\n", *ptr)
	if err != nil {
		fmt.Printf("Bad input: %s\n", input)
	}

}

```
The corrected version of `user` checks `err` before using `ptr`.


```go
package main

import (
	"fmt"
	"os"
)

func user(input string) {

	ptr, err := os.Open(input)
	if err != nil {
		fmt.Printf("Bad input: %s\n", input)
		return
	}
	// GOOD: `err` has been checked before `ptr` is used
	fmt.Printf("Result was %v\n", *ptr)

}

```

## References
* The Go Blog: [Error handling and Go](https://blog.golang.org/error-handling-and-go).
