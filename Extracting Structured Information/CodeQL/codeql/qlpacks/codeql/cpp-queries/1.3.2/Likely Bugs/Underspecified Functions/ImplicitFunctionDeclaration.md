# Implicit function declaration
A function is called without a prior function declaration or definition. When this happens, the compiler generates an implicit declaration of the function, specifying an integer return type and no parameters. If the implicit declaration does not match the true signature of the function, the function may behave unpredictably.

This may indicate a misspelled function name, or that the required header containing the function declaration has not been included.


## Recommendation
Provide an explicit declaration of the function before invoking it.


## Example

```c
/* '#include <stdlib.h>' was forgotten */

int main(void) {
	/* 'int malloc()' assumed */
	unsigned char *p = malloc(100);
	*p = 'a';
	return 0;
}
```

## References
* SEI CERT C Coding Standard: [DCL31-C. Declare identifiers before using them](https://wiki.sei.cmu.edu/confluence/display/c/DCL31-C.+Declare+identifiers+before+using+them)
