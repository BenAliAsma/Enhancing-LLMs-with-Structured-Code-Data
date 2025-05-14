# Impossible interface nil check
Interface values in Go are type tagged, that is, they are essentially pairs of the form `(value, type)`, where `value` is a non-interface value with the given `type`. Such a pair is never `nil`, even if `value` is `nil`.

In particular, if a non-interface value `v` is assigned to a variable `x` whose type is an interface, then `x` will never be `nil`, regardless of `v`. Comparing `x` to `nil` is pointless, and may indicate a misunderstanding of Go interface values or some other underlying bug.


## Recommendation
Carefully inspect the comparison to ensure it is not a symptom of a bug.


## Example
The following example shows a declaration of a function `fetch` which fetches the contents of a URL, returning either the contents or an error value, which is a pointer to a custom error type `RequestError` (not shown). The function `niceFetch` wraps this function, printing out either the URL contents or an error message.


```go
package main

import "fmt"

func fetch(url string) (string, *RequestError)

func niceFetch(url string) {
	var s string
	var e error
	s, e = fetch(url)
	if e != nil {
		fmt.Printf("Unable to fetch URL: %v\n", e)
	} else {
		fmt.Printf("URL contents: %s\n", s)
	}
}

```
However, because `e` is declared to be of type `error`, which is an interface, the `nil` check will never succeed, since `e` can never be `nil`.

In this case, the problem can be solved by using a short variable declaration using operator `:=`, which will automatically infer the more precise non-interface type `*ResourceError` for `e`, making the `nil` check behave as expected.


```go
package main

import "fmt"

func niceFetchGood(url string) {
	s, e := fetch(url)
	if e != nil {
		fmt.Printf("Unable to fetch URL: %v\n", e)
	} else {
		fmt.Printf("URL contents: %s\n", s)
	}
}

```

## References
* Jordan Orelli: [How to use interfaces in Go](https://jordanorelli.com/post/32665860244/how-to-use-interfaces-in-go).
* Go Frequently Asked Questions: [Why is my nil error value not equal to nil?](https://golang.org/doc/faq#nil_error).
