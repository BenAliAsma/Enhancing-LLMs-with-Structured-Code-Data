# Return stack-allocated object
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights return statements that return pointers to an object allocated on the stack. The lifetime of a stack allocated memory location only lasts until the function returns, , and the contents of that memory become undefined after that. Clearly, using a pointer to stack memory after the function has already returned will have undefined results.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Do not return pointers to stack memory locations. Instead, create an output parameter, or create a heap-allocated buffer, copy the contents of the stack allocated memory to that buffer and return that instead.


## Example

```cpp
Record* fixRecord(Record* r) {
	Record myRecord = *r;
	delete r;

	myRecord.fix();
	return &myRecord; //returns pointer to myRecord, which is a stack-allocated object
}

```

## References
* AV Rule 111, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
