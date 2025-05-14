# Function is never called
This query highlights functions that are non-public, non-virtual, and are never called. Dead functions are often deprecated pieces of code, and should be removed. If left in the code base they increase object code size, decrease code comprehensibility, and create the possibility of misuse.

`public` and `protected` functions are ignored by this query. This type of function may be part of the program's API and could be used by external programs.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute which function is actually called in a virtual call, or a call through a pointer, without running the program with all possible input data.

## Recommendation
Verify that the function is genuinely unused and consider removing it.


## Example
The example below includes a function `f` that is no longer used and should be deleted.


```cpp
class C {
public:
	void g() {
		...
		//f() was previously used but is now commented-out, orphaning f()
		//f();
		...
	}
private:
	void f() { //is now unused, and can be removed
	}
};

```

## References
* SEI CERT C++ Coding Standard: [MSC12-C. Detect and remove code that has no effect or is never executed](https://wiki.sei.cmu.edu/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
