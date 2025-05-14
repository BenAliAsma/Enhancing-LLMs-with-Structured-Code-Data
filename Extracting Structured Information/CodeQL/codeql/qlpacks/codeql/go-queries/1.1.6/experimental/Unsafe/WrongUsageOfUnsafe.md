# Wrong usage of package unsafe
The package \`unsafe\` provides operations that step outside the type safety guarantees normally provided inside Go programs. This leaves room for errors in the usage of the \`unsafe\` package that are not caught by the compiler.

One possible error is type casting between types of different sizes, that can result in reads to memory locations that are outside the bounds of the original target buffer, and also unexpected values.


## Recommendation
The basic recommendation is to avoid usage of the package \`unsafe\`. If that is not an option, you should always check the size of types you cast your data to/from to make sure they won't result in reading outside of the intended bounds.


## Example
The first example explores a few scenarios of read out of bounds because of an erroneous type casting done with `unsafe.Pointer`.


```go
package main

import (
	"fmt"
	"unsafe"
)

func main() {}

func badArrays() {
	// A harmless piece of data:
	harmless := [8]byte{'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'}
	// Something secret:
	secret := [9]byte{'s', 'e', 'n', 's', 'i', 't', 'i', 'v', 'e'}

	// The declared `leaking` contains data from `harmless`
	// plus the data from `secret`;
	// (notice we read more than the length of harmless)
	var leaking = (*[8 + 9]byte)(unsafe.Pointer(&harmless)) // BAD

	fmt.Println(string((*leaking)[:]))

	// Avoid optimization:
	if secret[0] == 123 {
		fmt.Println("hello world")
	}
}
func badIndexExpr() {
	// A harmless piece of data:
	harmless := [8]byte{'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'}
	// Something secret:
	secret := [9]byte{'s', 'e', 'n', 's', 'i', 't', 'i', 'v', 'e'}

	// Read before secret, overflowing into secret.
	// NOTE: unsafe.Pointer(&harmless) != unsafe.Pointer(&harmless[2])
	// Even tough harmless and leaking have the same size,
	// the new variable `leaking` will contain data starting from
	// the address of the 3rd element of the `harmless` array,
	// and continue for 8 bytes, going out of the boundaries of
	// `harmless` and crossing into the memory occupied by `secret`.
	var leaking = (*[8]byte)(unsafe.Pointer(&harmless[2])) // BAD

	fmt.Println(string((*leaking)[:]))

	// Avoid optimization:
	if secret[0] == 123 {
		fmt.Println("hello world")
	}
}

```
The second example show an example of correct type casting done with `unsafe.Pointer`.


```go
package main

import (
	"fmt"
	"unsafe"
)

func main() {}

func good0() {
	// A harmless piece of data:
	harmless := [8]byte{'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'}
	// Something secret:
	secret := [9]byte{'s', 'e', 'n', 's', 'i', 't', 'i', 'v', 'e'}

	// Read before secret without overflowing to secret;
	// the new `target` variable contains only data
	// inside the bounds of `harmless`, without overflowing
	// into the memory occupied by `secret`.
	var target = (*[8]byte)(unsafe.Pointer(&harmless)) // OK

	fmt.Println(string((*target)[:]))

	// Avoid optimization:
	if secret[0] == 123 {
		fmt.Println("hello world")
	}
}

```

## References
* [Go: What is the Unsafe Package?](https://medium.com/a-journey-with-go/go-what-is-the-unsafe-package-d2443da36350).
* [Go: Cast vs Conversion by Example](https://medium.com/a-journey-with-go/go-cast-vs-conversion-by-example-26e0ef3003f0).
* [Exploitation Exercise with unsafe.Pointer in Go: Information Leak (Part 1)](https://dev.to/jlauinger/exploitation-exercise-with-unsafe-pointer-in-go-information-leak-part-1-1kga).
* Common Weakness Enumeration: [CWE-119](https://cwe.mitre.org/data/definitions/119.html).
* Common Weakness Enumeration: [CWE-126](https://cwe.mitre.org/data/definitions/126.html).
