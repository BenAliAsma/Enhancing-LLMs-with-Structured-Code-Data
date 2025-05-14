# Return value of a function is ignored
This rule finds calls to a function that ignore the return value. A function call is only marked as a violation if at least 90% of the total calls to that function check the return value. Not checking a return value is a common source of defects from standard library functions like `malloc` or `fread`. These functions return the status information and the return values should always be checked to see if the operation succeeded before operating on any data modified or resources allocated by these functions.

This rule uses a white/blacklist of functions the value of which can always be ignored (e.g. `select`) and those that should always be checked (e.g. `fgets`). These list can be modified to suit a particular codebase.


## Recommendation
Check the return value of functions that return status information.


## Example

```cpp
int doFoo() {
	...
	return status;
}

void f() {
	if (doFoo() == OK) {
		...
	}
}

void g() {
	int status = doFoo();
	if (status == OK) {
		...
	}
}

void err() {
	doFoo(); //doFoo is called but its return value is not checked, and 
	         //the value is checked in other locations
	...
}

```

## References
* M. Henricson and E. Nyquist, *Industrial Strength C++*, Chapter 12: Error handling. Prentice Hall PTR, 1997 ([available online](https://web.archive.org/web/20190919025638/https://mongers.org/industrial-c++/)).
* The CERT C Secure Coding Standard: [EXP32-PL. Do not ignore function return values](https://www.securecoding.cert.org/confluence/display/perl/EXP32-PL.+Do+not+ignore+function+return+values).
* Common Weakness Enumeration: [CWE-252](https://cwe.mitre.org/data/definitions/252.html).
