# Pointer to stack object used as return value
This query finds return statements that return pointers to an object allocated on the stack. The lifetime of a stack allocated memory location only lasts until the function returns, and the contents of that memory become undefined after that. Clearly, using a pointer to stack memory after the function has already returned will have undefined results.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Do not return pointers to stack memory locations. Instead, create an output parameter, or create a heap-allocated buffer. You can then copy the contents of the stack-allocated memory to the heap-allocated buffer and return that location instead.


## Example
The example below the reference to `myRecord` is useful only while the containing function is running. If you need to access the object outside this function, either create an output parameter with its value, or copy the object into heap-allocated memory.


```cpp
Record* fixRecord(Record* r) {
	Record myRecord = *r;
	delete r;

	myRecord.fix();
	return &myRecord; //returns reference to myRecord, which is a stack-allocated object
}

```

## References
* cplusplus.com: [Pointers](http://www.cplusplus.com/doc/tutorial/pointers/).
* The craft of coding: [Memory in C - the stack, the heap, and static](https://craftofcoding.wordpress.com/2015/12/07/memory-in-c-the-stack-the-heap-and-static/).
* Common Weakness Enumeration: [CWE-562](https://cwe.mitre.org/data/definitions/562.html).
