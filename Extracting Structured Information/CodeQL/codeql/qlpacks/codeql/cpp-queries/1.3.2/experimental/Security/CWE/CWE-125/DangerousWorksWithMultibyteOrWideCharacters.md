# Dangerous use convert function.
Using a function to convert multibyte or wide characters with an invalid length argument may result in an out-of-range access error or unexpected results.


## Example
The following example shows the erroneous and corrected method of using function mbtowc.


```cpp

...
mbtowc(&wc, ptr, 4)); // BAD:we can get unpredictable results
...
mbtowc(&wc, ptr, MB_LEN_MAX); // GOOD
...

```

## References
* CERT Coding Standard: [ARR30-C. Do not form or use out-of-bounds pointers or array subscripts - SEI CERT C Coding Standard - Confluence](https://wiki.sei.cmu.edu/confluence/display/c/ARR30-C.+Do+not+form+or+use+out-of-bounds+pointers+or+array+subscripts).
* Common Weakness Enumeration: [CWE-125](https://cwe.mitre.org/data/definitions/125.html).
