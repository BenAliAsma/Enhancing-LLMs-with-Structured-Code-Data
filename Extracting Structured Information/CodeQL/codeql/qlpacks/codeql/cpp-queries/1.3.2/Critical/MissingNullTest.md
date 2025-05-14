# Returned pointer not checked
This query finds pointer dereferences that use a pointer returned from a function which may return NULL. Always check your pointers for NULL-ness before dereferencing them. Dereferencing a null pointer and attempting to modify its contents can lead to anything from a segmentation fault to corruption of important system data (for example, the interrupt table in some architectures).


## Recommendation
Add a null check before dereferencing the pointer, or modify the function so that it always returns a non-null value.


## Example
In this example, the function is not protected from dereferencing a null pointer. It should be updated to ensure that this cannot happen.


```cpp
typedef struct {
	char name[100];
	int status;
} person;

void f() {
	person* buf = NULL;
	buf = malloc(sizeof(person));

	(*buf).status = 0; //access to buf before it was checked for NULL
}

```

## References
* SEI CERT C Coding Standard: [EXP34-C. Do not dereference null pointers](https://wiki.sei.cmu.edu/confluence/display/c/EXP34-C.+Do+not+dereference+null+pointers).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
