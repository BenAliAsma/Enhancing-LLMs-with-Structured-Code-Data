# Potentially unsafe call to strncat
The standard library function `strncat` appends a source string to a target string. The third argument defines the maximum number of characters to append and should be less than or equal to the remaining space in the destination buffer.

Calls of the form `strncat(dest, src, strlen(dest))` or `strncat(dest, src, sizeof(dest))` set the third argument to the entire size of the destination buffer. Executing a call of this type may cause a buffer overflow unless the buffer is known to be empty.

Similarly, calls of the form `strncat(dest, src, sizeof (dest) - strlen (dest))` allow one byte to be written outside the `dest` buffer.

Buffer overflows can lead to anything from a segmentation fault to a security vulnerability.


## Recommendation
Check the highlighted function calls carefully to ensure that no buffer overflow is possible. For a more robust solution, consider updating the function call to include the remaining space in the destination buffer.


## Example

```cpp
strncat(dest, src, strlen(dest)); //wrong: should use remaining size of dest

strncat(dest, src, sizeof(dest)); //wrong: should use remaining size of dest. 
                                  //Also fails if dest is a pointer and not an array.
 
strncat(dest, source, sizeof(dest) - strlen(dest)); // wrong: writes a zero byte past the `dest` buffer.

strncat(dest, source, sizeof(dest) - strlen(dest) - 1); // correct: reserves space for the zero byte.

```

## References
* cplusplus.com: [strncat](http://www.cplusplus.com/reference/clibrary/cstring/strncat/), [strncpy](http://www.cplusplus.com/reference/clibrary/cstring/strncpy/).
* I. Gerg, *An Overview and Example of the Buffer-Overflow Exploit*. IANewsletter vol 7 no 4, 2005.
* M. Donaldson, *Inside the Buffer Overflow Attack: Mechanism, Method &amp; Prevention*. SANS Institute InfoSec Reading Room, 2002.
* CERT C Coding Standard: [STR31-C. Guarantee that storage for strings has sufficient space for character data and the null terminator](https://wiki.sei.cmu.edu/confluence/display/c/STR31-C.+Guarantee+that+storage+for+strings+has+sufficient+space+for+character+data+and+the+null+terminator).
* Common Weakness Enumeration: [CWE-788](https://cwe.mitre.org/data/definitions/788.html).
* Common Weakness Enumeration: [CWE-676](https://cwe.mitre.org/data/definitions/676.html).
* Common Weakness Enumeration: [CWE-119](https://cwe.mitre.org/data/definitions/119.html).
* Common Weakness Enumeration: [CWE-251](https://cwe.mitre.org/data/definitions/251.html).
