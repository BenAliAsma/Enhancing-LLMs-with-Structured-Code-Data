# Missing return statement
This rule finds non-void functions with an execution path that does not return through an explicit return statement. The return value in such a case is undefined. For example, in the `cdecl` calling convention for x86, it would be whatever value was in the AX/EAX register when the function returned, assuming the function had a non-float return type that can fit in a machine word.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
Make sure that all execution paths in the function exit through an explicit return statement.


## Example

```cpp
int f() {
	...
	if (error) {
		return -1;
	}
	...
	//wrong: no explicit return here, value returned is undefined
}

```

## References
* AV Rule 114, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* MISRA C++ Rule 8-4-3, *Guidelines for the use of the C++ language in critical systems*. The Motor Industry Software Reliability Associate, 2008.
* MSDN Library: [return Statement (C++)](https://docs.microsoft.com/en-us/cpp/cpp/return-statement-cpp).
