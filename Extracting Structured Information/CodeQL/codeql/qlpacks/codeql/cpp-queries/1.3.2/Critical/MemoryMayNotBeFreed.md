# Memory may not be freed
This rule looks for functions that allocate memory, but may return without freeing it. This can occur when an operation performed on the memory block fails, and the function returns with an error before freeing the allocated block. This causes the function to leak memory and may eventually lead to software failure.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
Ensure that the function frees all dynamically allocated memory it has acquired in all circumstances, unless that memory is returned to the caller.


## Example

```cpp
int* f() {
	try {
		int *buff = malloc(SIZE*sizeof(int));
		do_stuff(buff);
		return buff;
	} catch (int do_stuff_exception) {
		return NULL; //returns NULL on error, but does not free memory
	}
}

```
In this example, if an exception occurs the memory allocated into `buff` is neither freed or returned. To fix this memory leak, we could add code to free `buff` to the `catch` block as follows:


```cpp
int* f() {
	int *buff = NULL;
	try {
		buff = malloc(SIZE*sizeof(int));
		do_stuff(buff);
		return buff;
	} catch (int do_stuff_exception) {
		if (buff != NULL) {
			free(buff);
		}
		return NULL; //returns NULL on error, having freed any allocated memory
	}
}

```
