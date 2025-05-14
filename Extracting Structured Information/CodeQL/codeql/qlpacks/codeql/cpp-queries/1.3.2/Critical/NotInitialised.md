# Variable not initialized before use
This query finds variables that are used before they are initialized. Values of uninitialized variables are undefined, as not all compilers zero out memory, especially when optimizations are enabled or the compiler is not compliant with the latest language standards.


## Recommendation
Initialize the variable before accessing it.


## Example

```cpp
{
	int i;

	...
	int g = COEFF * i; //i is used before it is initialized
}

```

## References
* C++ reference: [uninitialized variables](https://en.cppreference.com/book/uninitialized).
* Common Weakness Enumeration: [CWE-457](https://cwe.mitre.org/data/definitions/457.html).
