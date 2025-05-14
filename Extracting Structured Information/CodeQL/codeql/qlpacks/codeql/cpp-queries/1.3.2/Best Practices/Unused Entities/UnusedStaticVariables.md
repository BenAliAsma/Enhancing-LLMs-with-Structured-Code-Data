# Unused static variable
This rule finds static variables that are never accessed. These static variables should be removed, to increase code comprehensibility, reduce memory usage and avoid misuse. Unused static `const` variables could also be an indication of a defect caused by an unhandled case.


## Recommendation
Check that the unused static variable does not indicate a defect, for example, an unhandled case. If the static variable is genuinely not needed, then removing it will make code more readable. If the static variable is needed then you should update the code to fix the defect.


## Example

```cpp
void f() {
	static int i = 0; //i is unused
	...
	return;
}

```

## References
* [Variable scope](http://www.tutorialspoint.com/cplusplus/cpp_variable_scope.htm)
* [Detect and remove code that has no effect](https://www.securecoding.cert.org/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed)
* [Minimize the scope of variables and functions](https://wiki.sei.cmu.edu/confluence/display/c/DCL19-C.+Minimize+the+scope+of+variables+and+functions)
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
