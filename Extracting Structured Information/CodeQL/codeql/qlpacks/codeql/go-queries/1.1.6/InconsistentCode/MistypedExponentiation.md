# Bitwise exclusive-or used like exponentiation
The caret symbol (`^`) is sometimes used to represent exponentiation but in Go, as in many C-like languages, it represents the bitwise exclusive-or operation. The expression as `2^32` thus evaluates the number 34, not 2<sup>32</sup>, and it is likely that patterns such as this are mistakes.


## Recommendation
To compute 2<sup><code>EXP</code></sup>, `1 << EXP` can be used. For constant exponents, `1eEXP` can be used to find 10<sup><code>EXP</code></sup>. In other cases, there is `math.Pow` in the Go standard library which provides this functionality.


## Example
The example below prints 34 and not 2<sup>32</sup> (4294967296).


```go
package main

import "fmt"

func main() {
	fmt.Println(2 ^ 32) // should be 1 << 32
}

```

## References
* GCC Bugzilla: [GCC should warn about 2^16 and 2^32 and 2^64](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90885)
