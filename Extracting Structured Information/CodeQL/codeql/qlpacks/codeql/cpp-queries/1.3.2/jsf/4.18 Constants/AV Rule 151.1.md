# Constant string literals
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights string literals that are assigned to a non-`const` variable. String literals should not be changed, since they are usually stored in the data section, and depending on the architecture, writing to the data section will cause undefined behavior, such as memory corruption or memory write error.


## Recommendation
Only assign string literals to `const` variables. In general, using `const` to indicate values that do not change is good practice, as it provides a compile-time check and when used on function parameters gives an indication of the function's expected behavior.


## Example

```cpp
void f() {
	char *s = "String"; //wrong: literal assigned to non-const
	//this will cause a write error or corrupt other data in the data section
	strcpy(s, "Another string"); 

	const char* cs ="String"; //correct: literal assigned to a const
	//this will cause a compile error (trying to write to a const)
	strcpy(cs, "Another string");
}

```

## References
* AV Rule 151.1, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
