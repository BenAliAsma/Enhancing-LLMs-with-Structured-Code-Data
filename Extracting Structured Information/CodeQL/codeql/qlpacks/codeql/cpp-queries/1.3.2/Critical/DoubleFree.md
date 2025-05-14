# Potential double free
Deallocating memory more than once can lead to a double-free vulnerability. This can be exploited to corrupt the allocator's internal data structures, which can lead to denial-of-service attacks by crashing the program, or security vulnerabilities, by allowing an attacker to overwrite arbitrary memory locations.


## Recommendation
Ensure that all execution paths deallocate the allocated memory at most once. In complex cases it may help to reassign a pointer to a null value after deallocating it. This will prevent double-free vulnerabilities since most deallocation functions will perform a null-pointer check before attempting to deallocate memory.


## Example
In the following example, `buff` is allocated and then freed twice:


```cpp
int* f() {
	int *buff = malloc(SIZE*sizeof(int));
	do_stuff(buff);
	free(buff);
	int *new_buffer = malloc(SIZE*sizeof(int));
	free(buff); // BAD: If new_buffer is assigned the same address as buff,
              // the memory allocator will free the new buffer memory region,
              // leading to use-after-free problems and memory corruption.
	return new_buffer;
}

```
Reviewing the code above, the issue can be fixed by simply deleting the additional call to `free(buff)`.


```cpp
int* f() {
	int *buff = malloc(SIZE*sizeof(int));
	do_stuff(buff);
	free(buff); // GOOD: buff is only freed once.
	int *new_buffer = malloc(SIZE*sizeof(int));
	return new_buffer;
}

```
In the next example, `task` may be deleted twice, if an exception occurs inside the `try` block after the first `delete`:


```cpp
void g() {
	MyTask *task = nullptr;

	try
	{
		task = new MyTask;

		...

		delete task;

		...
	} catch (...) {
		delete task; // BAD: potential double-free
	}
}

```
The problem can be solved by assigning a null value to the pointer after the first `delete`, as calling `delete` a second time on the null pointer is harmless.


```cpp
void g() {
	MyTask *task = nullptr;

	try
	{
		task = new MyTask;

		...

		delete task;
		task = nullptr;

		...
	} catch (...) {
		delete task; // GOOD: harmless if task is NULL
	}
}

```

## References
* OWASP: [Doubly freeing memory](https://owasp.org/www-community/vulnerabilities/Doubly_freeing_memory).
* Common Weakness Enumeration: [CWE-415](https://cwe.mitre.org/data/definitions/415.html).
