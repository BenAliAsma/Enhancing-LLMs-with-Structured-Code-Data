# Open file may not be closed
This rule looks at functions that return a `FILE*`, but may return an error value before actually closing the resource. This can occur when an operation performed on the open descriptor fails, and the function returns with an error before closing the open resource. An improperly handled error may cause the function to leak file descriptors.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
Ensure that the function frees all the resources it acquired when an error occurs.


## Example

```cpp
FILE* f() {
	try {
		FILE *fp = fopen("foo.txt", "w");
		do_stuff(fp);
		return fp; //if there are no exceptions, the file pointer is returned correctly
	} catch (int do_stuff_exception) {
		return NULL; //returns NULL on error, but does not close fp
	}
}

```
