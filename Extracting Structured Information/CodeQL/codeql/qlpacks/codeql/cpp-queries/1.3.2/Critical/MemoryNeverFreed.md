# Memory is never freed
This rule finds calls to the `alloc` family of functions without a corresponding `free` call in the entire program. This leads to memory leaks.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the exact value of the variable without running the program with all possible input data.

## Recommendation
Ensure that all memory allocated by the program is freed before it terminates.


## Example

```cpp
int main(int argc, char* argv[]) {
	int *buff = malloc(SIZE * sizeof(int));
	int status = 0;
	... //code that does not free buff
	return status; //buff is never closed
}

```
