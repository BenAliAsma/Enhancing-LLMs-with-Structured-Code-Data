# Likely overrunning write
The program performs a buffer copy or write operation with no upper limit on the size of the copy. By analyzing the bounds of the expressions involved, it appears that certain inputs will cause a buffer overflow to occur in this case. In addition to causing program instability, techniques exist which may allow an attacker to use this vulnerability to execute arbitrary code.


## Recommendation
Always control the length of buffer copy and buffer write operations. `strncpy` should be used over `strcpy`, `snprintf` over `sprintf`, and in other cases 'n-variant' functions should be preferred.


## Example

```c
int sayHello(uint32_t userId)
{
	char buffer[17];

	if (userId > 9999) return USER_ID_OUT_OF_BOUNDS;

	// BAD: this message overflows the buffer if userId >= 1000,
	// as no space for the null terminator was accounted for
	sprintf(buffer, "Hello, user %d!", userId);

	MessageBox(hWnd, buffer, "New Message", MB_OK);
	
	return SUCCESS;
}
```
In this example, the call to `sprintf` writes a message of 14 characters (including the terminating null) plus the length of the string conversion of \`userId\` into a buffer with space for just 17 characters. While \`userId\` is checked to occupy no more than 4 characters when converted, there is no space in the buffer for the terminating null character if \`userId &gt;= 1000\`. In this case, the null character overflows the buffer resulting in undefined behavior.

To fix this issue these changes should be made:

* Control the size of the buffer by declaring it with a compile time constant.
* Preferably, replace the call to `sprintf` with `snprintf`, using the defined constant size of the buffer or \`sizeof(buffer)\` as maximum length to write. This will prevent the buffer overflow.
* Increasing the buffer size to account for the full range of \`userId\` and the terminating null character.

## References
* CERT C Coding Standard: [STR31-C. Guarantee that storage for strings has sufficient space for character data and the null terminator](https://www.securecoding.cert.org/confluence/display/c/STR31-C.+Guarantee+that+storage+for+strings+has+sufficient+space+for+character+data+and+the+null+terminator).
* CERT C++ Coding Standard: [STR50-CPP. Guarantee that storage for strings has sufficient space for character data and the null terminator](https://www.securecoding.cert.org/confluence/display/cplusplus/STR50-CPP.+Guarantee+that+storage+for+strings+has+sufficient+space+for+character+data+and+the+null+terminator).
* Common Weakness Enumeration: [CWE-120](https://cwe.mitre.org/data/definitions/120.html).
* Common Weakness Enumeration: [CWE-787](https://cwe.mitre.org/data/definitions/787.html).
* Common Weakness Enumeration: [CWE-805](https://cwe.mitre.org/data/definitions/805.html).
