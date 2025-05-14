# No space for zero terminator
This rule identifies calls to `malloc` that call `strlen` to determine the required buffer size, but do not allocate space for the zero terminator.


## Recommendation
The highlighted code segment creates a buffer without ensuring it's large enough to accommodate the copied data. This leaves the code susceptible to a buffer overflow attack, which could lead to anything from program crashes to malicious code execution.

Increase the size of the buffer being allocated by one or replace `malloc`, `strcpy` pairs with a call to `strdup`


## Example

```c

void flawed_strdup(const char *input)
{
	char *copy;

	/* Fail to allocate space for terminating '\0' */
	copy = (char *)malloc(strlen(input));
	strcpy(copy, input);
	return copy;
}


```

## References
* CERT C Coding Standard: [MEM35-C. Allocate sufficient memory for an object](https://www.securecoding.cert.org/confluence/display/c/MEM35-C.+Allocate+sufficient+memory+for+an+object).
* Common Weakness Enumeration: [CWE-131](https://cwe.mitre.org/data/definitions/131.html).
* Common Weakness Enumeration: [CWE-120](https://cwe.mitre.org/data/definitions/120.html).
* Common Weakness Enumeration: [CWE-122](https://cwe.mitre.org/data/definitions/122.html).
