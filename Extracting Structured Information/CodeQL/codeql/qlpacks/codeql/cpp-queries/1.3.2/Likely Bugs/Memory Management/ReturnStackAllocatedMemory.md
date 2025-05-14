# Returning stack-allocated memory
This rule finds return statements that return pointers to an object allocated on the stack. The lifetime of a stack allocated memory location only lasts until the function returns, and the contents of that memory become undefined after that. Clearly, using a pointer to stack memory after the function has already returned will have undefined results.


## Recommendation
Use the functions of the `malloc` family, or `new`, to dynamically allocate memory on the heap for data that is used across function calls.


## Example
The following example allocates an object on the stack and returns a pointer to it. This is incorrect because the object is deallocated when the function returns, and the pointer becomes invalid.


```cpp
Record *mkRecord(int value) {
	Record myRecord(value);

	return &myRecord; // BAD: returns a pointer to `myRecord`, which is a stack-allocated object.
}

```
To fix this, allocate the object on the heap using `new` and return a pointer to the heap-allocated object.


```cpp
Record *mkRecord(int value) {
	Record *myRecord = new Record(value);

	return myRecord; // GOOD: returns a pointer to a `myRecord`, which is a heap-allocated object.
}

```

## References
* Common Weakness Enumeration: [CWE-825](https://cwe.mitre.org/data/definitions/825.html).
