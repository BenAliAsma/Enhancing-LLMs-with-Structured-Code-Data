# Copy function using source size
The bounded copy functions `memcpy`, `memmove`, `strncpy`, `strncat` accept a size argument. You should call these functions with a size argument that is derived from the size of the destination buffer. Using a size argument that is derived from the source buffer may cause a buffer overflow. Buffer overflows can lead to anything from a segmentation fault to a security vulnerability.


## Recommendation
Check the highlighted function calls carefully. Ensure that the size parameter is derived from the size of the destination buffer, and not the source buffer.

> WARNING: This check is an approximation, so some results may not be actual defects in the program. It is not possible in general to compute the exact value of the variable without running the program with all possible input data.

## Example
The code below shows an example where `strncpy` is called incorrectly, without checking the size of the destination buffer. In the second example the call has been updated to include the size of the destination buffer.


```cpp

int main(int argc, char* argv[]) {
	char param[20];
	char *arg1;

	arg1 = argv[1];

	//wrong: only uses the size of the source (argv[1]) when using strncpy
	strncpy(param, arg1, strlen(arg1));

	//correct: uses the size of the destination array as well
	strncpy(param, arg1, min(strlen(arg1), sizeof(param) -1));
}

```

## References
* I. Gerg. *An Overview and Example of the Buffer-Overflow Exploit*. IANewsletter vol 7 no 4. 2005.
* M. Donaldson. *Inside the Buffer Overflow Attack: Mechanism, Method &amp; Prevention*. SANS Institute InfoSec Reading Room. 2002.
* Common Weakness Enumeration: [CWE-119](https://cwe.mitre.org/data/definitions/119.html).
* Common Weakness Enumeration: [CWE-131](https://cwe.mitre.org/data/definitions/131.html).
