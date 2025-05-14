# Global variable may be used before initialization
This rule finds calls to functions that use a global variable before the variable has been initialized. Not all compilers generate code that zero-out memory, especially when optimizations are enabled or the compiler is not compliant with the latest language standards. Accessing uninitialized memory will lead to undefined results.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the actual branch taken in conditional statements such as "if" without running the program with all possible input data. This means that it is not possible to determine if a particular statement is going to be executed.

## Recommendation
Initialize the global variable. If no constant can be used for initialization, ensure that all accesses to the variable occur after the initialization code is executed.


## Example
In the example below, `callCtr` is wrongly used before it has been initialized.


```cpp
int g_callCtr;

void initGlobals() {
	g_callCtr = 0;
}

int main(int argc, char* argv[]) {
	...
	cout << g_callCtr; //callCtr used before it is initialized
	initGlobals();
}

```

## References
* SEI CERT C++ Coding Standard: [EXP53-CPP. Do not read uninitialized memory](https://wiki.sei.cmu.edu/confluence/display/cplusplus/EXP53-CPP.+Do+not+read+uninitialized+memory).
* Common Weakness Enumeration: [CWE-457](https://cwe.mitre.org/data/definitions/457.html).
