# Potential buffer overflow
This rule highlights potentially overflowing calls to the functions `sprintf` and `vsprintf` with a warning. These functions allow unbounded writes to buffers, which may cause an overflow when used on untrusted data or without adequate checks on the size of the data. Function calls of this type constitute a security risk through buffer overflows.


## Recommendation
Always control the length of buffer copy and buffer write operations. Use the safer variants `snprintf` and `vsnprintf`, which include an extra buffer length argument.


## Example

```cpp
void f(char* s, float f) {
	char buf[30];

	//wrong: gets has no limit to the length of data it puts in the buffer
	gets(buf); 

	//wrong: sprintf does not limit the length of the string put into buf
	sprintf(buf, "This is a string: %s", s);

	//wrong: %f can expand to a very long string in extreme cases, easily overrunning this buffer
	sprintf(buf, "This is a float: %f", f);
}

```
To improve the security of this example code, three changes should be made:

1. Introduce a preprocessor define for the size of the buffer.
1. Replace both calls to `sprintf` with `snprintf`, specifying the define as the maximum length to copy. This will prevent the buffer overflow.
1. Consider using the %g format specifier instead of %f.

## References
* Common Weakness Enumeration: [CWE-120: Buffer Copy without Checking Size of Input ('Classic Buffer Overflow').](http://cwe.mitre.org/data/definitions/120.html)
* CERT C Coding Standard: [STR31-C. Guarantee that storage for strings has sufficient space for character data and the null terminator](https://www.securecoding.cert.org/confluence/display/c/STR31-C.+Guarantee+that+storage+for+strings+has+sufficient+space+for+character+data+and+the+null+terminator).
* M. Howard, D. Leblanc, J. Viega, *19 Deadly Sins of Software Security: Programming Flaws and How to Fix Them*, McGraw-Hill Osborne, 2005.
* Common Weakness Enumeration: [CWE-676](https://cwe.mitre.org/data/definitions/676.html).
