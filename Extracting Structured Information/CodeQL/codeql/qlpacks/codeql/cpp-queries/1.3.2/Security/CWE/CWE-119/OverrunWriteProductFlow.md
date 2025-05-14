# Overrunning write
You must ensure that you do not exceed the size of an allocation during write and read operations. If an operation attempts to write to or access an element that is outside the range of the allocation then this results in a buffer overflow. Buffer overflows can lead to anything from a segmentation fault to a security vulnerability.


## Recommendation
Check the offsets and sizes used in the highlighted operations to ensure that a buffer overflow will not occur.


## Example

```cpp
int f(char * s, unsigned size) {
	char* buf = (char*)malloc(size);

	strncpy(buf, s, size + 1); // wrong: copy may exceed size of buf

	for (int i = 0; i <= size; i++) { // wrong: upper limit that is higher than size of buf
		cout << buf[i];
	}
}

```

## References
* I. Gerg. *An Overview and Example of the Buffer-Overflow Exploit*. IANewsletter vol 7 no 4. 2005.
* M. Donaldson. *Inside the Buffer Overflow Attack: Mechanism, Method &amp; Prevention*. SANS Institute InfoSec Reading Room. 2002.
* Common Weakness Enumeration: [CWE-119](https://cwe.mitre.org/data/definitions/119.html).
* Common Weakness Enumeration: [CWE-131](https://cwe.mitre.org/data/definitions/131.html).
