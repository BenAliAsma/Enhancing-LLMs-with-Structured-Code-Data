# Unreachable code
This rule finds code that is never reached. Unused code should be removed to increase readability and avoid confusion.


## Recommendation
Remove any unreachable code.


## Example
In the following example, the final `return` statement can never be reached:


```rust
fn fib(input: u32) -> u32 {
	if (input == 0) {
		return 0;
	} else if (input == 1) {
		return 1;
	} else {
		return fib(input - 1) + fib(input - 2);
	}

	return input; // BAD: this code is never reached
}

```
The problem can be fixed simply by removing the unreachable code:


```rust
fn fib(input: u32) -> u32 {
	if (input == 0) {
		return 0;
	} else if (input == 1) {
		return 1;
	} else {
		return fib(input - 1) + fib(input - 2);
	}
}

```

## References
* Wikipedia: [Unreachable code](https://en.wikipedia.org/wiki/Unreachable_code)
