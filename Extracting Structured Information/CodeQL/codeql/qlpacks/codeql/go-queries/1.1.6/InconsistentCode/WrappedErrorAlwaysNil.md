# Wrapped error is always nil
The `pkg.errors` package provides the `errors.Wrap` function for annotating an error with a stack trace. When passed `nil`, this function returns `nil`. When the first parameter to `errors.Wrap` is *always* `nil`, the function call has no effect and likely indicates a programming mistake. A common example of this is when an `errors.Wrap(err, "message")` call is copied from an earlier error-handling block in the same function and used in a subsequent error-handling block that does not check `err` in its guard. In this case the return of a `nil` value to the caller indicates by convention that the operation succeeded, and so the failure is masked.


## Recommendation
Usually an `err` value is being referenced outside its intended scope. The problem can be fixed by removing that reference, for example by changing a call of the form `errors.Wrap(err, "message")` to `errors.New("message")`.


## Example
The example below shows an example where the `err` value returned from the call to `f1` is reused in a later call, when it is known to be `nil`:


```go
package main

import (
	"github.com/pkg/errors"
)

func f1(input string) error {
	if input == "1" {
		return errors.Errorf("error in f1")
	}
	return nil
}

func f2(input string) (bool, error) {
	if input == "2" {
		return false, errors.Errorf("error in f2")
	}
	return true, nil
}

func test1(input string) error {
	err := f1(input)
	if err != nil {
		return errors.Wrap(err, "input is the first non-negative integer")
	}
	if ok2, _ := f2(input); !ok2 {
		return errors.Wrap(err, "input is the second non-negative integer")
	}
	return nil
}

```
One way of fixing this is to create a new error value with `errors.New`:


```go
package main

import (
	"github.com/pkg/errors"
)

func f1(input string) error {
	if input == "1" {
		return errors.Errorf("error in f1")
	}
	return nil
}

func f2(input string) (bool, error) {
	if input == "2" {
		return false, errors.Errorf("error in f2")
	}
	return true, nil
}

func test1(input string) error {
	err := f1(input)
	if err != nil {
		return errors.Wrap(err, "input is the first non-negative integer")
	}
	if ok2, _ := f2(input); !ok2 {
		return errors.New("input is the second non-negative integer")
	}
	return nil
}

```

## References
* Go errors, github.com/pkg/errors: [errors.Wrap](https://pkg.go.dev/github.com/pkg/errors#Wrap)
