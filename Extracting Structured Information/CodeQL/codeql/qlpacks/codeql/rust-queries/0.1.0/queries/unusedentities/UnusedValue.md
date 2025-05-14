# Unused value
This rule finds values that are assigned to variables but never used. Unused values should be removed to increase readability and avoid confusion.


## Recommendation
Remove any unused values. Also remove any variables that only hold unused values.


## Example
In the following example, there is a variable `average` that is initialized to `0`, but that value is never used:


```rust
fn get_average(values:&[i32]) -> f64 {
	let mut sum = 0;
	let mut average = 0.0; // BAD: unused value

	for v in values {
		sum += v;
	}

	average = sum as f64 / values.len() as f64;
	return average;
}

```
The problem can be fixed by removing the unused value:


```rust
fn get_average(values:&[i32]) -> f64 {
	let mut sum = 0;
	let average;

	for v in values {
		sum += v;
	}

	average = sum as f64 / values.len() as f64;
	return average;
}

```

## References
* GeeksforGeeks: [How to avoid unused Variable warning in Rust?](https://www.geeksforgeeks.org/how-to-avoid-unused-variable-warning-in-rust/)
