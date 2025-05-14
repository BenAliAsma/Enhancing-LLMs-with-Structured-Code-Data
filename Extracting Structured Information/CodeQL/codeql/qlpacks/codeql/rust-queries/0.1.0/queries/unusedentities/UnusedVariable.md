# Unused variable
This rule finds variables that are never accessed. Unused variables should be removed to increase readability and avoid confusion.


## Recommendation
Remove any unused variables.


## Example
In the following example, there is an unused variable `average` that is never used:


```rust
fn get_sum(values:&[i32]) -> i32 {
	let mut sum = 0;
	let mut average; // BAD: unused variable

	for v in values {
		sum += v;
	}

	return sum;
}

```
The problem can be fixed simply by removing the variable:


```rust
fn get_sum(values:&[i32]) -> i32 {
	let mut sum = 0;

	for v in values {
		sum += v;
	}

	return sum;
}

```

## References
* GeeksforGeeks: [How to avoid unused Variable warning in Rust?](https://www.geeksforgeeks.org/how-to-avoid-unused-variable-warning-in-rust/)
