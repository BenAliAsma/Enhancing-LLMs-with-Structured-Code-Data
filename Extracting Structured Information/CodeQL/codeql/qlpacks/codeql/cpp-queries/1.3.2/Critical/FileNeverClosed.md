# Open file is not closed
This rule finds calls to `fopen` with no corresponding `fclose` call in the entire program. Leaving files open will cause a resource leak that will persist even after the program terminates.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the exact value of the variable without running the program with all possible input data.

## Recommendation
Ensure that all file or socket descriptors allocated by the program are freed before it terminates.


## Example

```cpp
int main(int argc, char* argv[]) {
	FILE *fp = fopen("foo.txt", "w");
	int status = 0;
	... //code that does not close fp
	return status; //fp is never closed
}

```
