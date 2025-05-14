# Buffer access with incorrect length value
Using a size argument that is larger than the buffer size will result in an out-of-bounds memory access and possibly overflow. You need to limit the value of the length argument.


## Example
The following example shows the use of a function with and without an error in the size argument.


```cpp

...
char buf[256];
X509_NAME_oneline(X509_get_subject_name(peer),buf,sizeof(buf)); // GOOD
...
char buf[256];
X509_NAME_oneline(X509_get_subject_name(peer),buf,1024); // BAD
...

```

## References
* CERT Coding Standard: [ARR38-C. Guarantee that library functions do not form invalid pointers - SEI CERT C Coding Standard - Confluence](https://wiki.sei.cmu.edu/confluence/display/c/ARR38-C.+Guarantee+that+library+functions+do+not+form+invalid+pointers).
* Common Weakness Enumeration: [CWE-805](https://cwe.mitre.org/data/definitions/805.html).
