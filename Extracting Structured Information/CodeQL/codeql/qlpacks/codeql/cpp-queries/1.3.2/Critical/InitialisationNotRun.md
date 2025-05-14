# Initialization code not run
The query finds code that initializes a global variable (that is, uses it as an L-value) but is never called from `main`. Unless there is another entry point that triggers the initialization, the code should be modified so that the variable is initialized. Uninitialized variables may contain any value, as not all compilers generate code that zero-out memory, especially when optimizations are enabled or the compiler is not compliant with the latest language standards.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute which function is actually called in a virtual call, or a call through a pointer, without running the program with all possible input data.

## Recommendation
Examine the code and ensure that the variable is always initialized before it is used.


## Example
In the example below, the code that triggers the initialization of `g_storage` is not run from `main`. Unless the variable is initialized by another method, the call on line 10 may dereference a null pointer.


```cpp
GlobalStorage *g_storage;

void init() { //initializes g_storage, but is never run from main
	g_storage = new GlobalStorage();
	...
}

int main(int argc, char *argv[]) {
	... //init not called
	strcpy(g_storage->name, argv[1]); // g_storage is used before init() is called
	...
}

```

## References
* C++ reference: [uninitialized variables](https://en.cppreference.com/book/uninitialized).
* Common Weakness Enumeration: [CWE-456](https://cwe.mitre.org/data/definitions/456.html).
