# Potential use after free
This rule finds accesses through a pointer of a memory location that has already been freed (i.e. through a dangling pointer). Such memory blocks have already been released to the dynamic memory manager, and modifying them can lead to anything from a segfault to memory corruption that would cause subsequent calls to the dynamic memory manager to behave erratically, to a possible security vulnerability.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the values of pointers without running the program with all input data.

## Recommendation
Ensure that all execution paths that access memory through a pointer never access that pointer after it is freed.


## Example

```cpp
void f() {
	char* buf = new char[SIZE];
	...
	if (error) {
		delete buf; //error handling has freed the buffer
	}
	...
	log_contents(buf); //but it is still used here for logging
	...
}

```

## References
* I. Gerg. *An Overview and Example of the Buffer-Overflow Exploit*. IANewsletter vol 7 no 4. 2005.
* M. Donaldson. *Inside the Buffer Overflow Attack: Mechanism, Method &amp; Prevention*. SANS Institute InfoSec Reading Room. 2002.
* Common Weakness Enumeration: [CWE-416](https://cwe.mitre.org/data/definitions/416.html).
