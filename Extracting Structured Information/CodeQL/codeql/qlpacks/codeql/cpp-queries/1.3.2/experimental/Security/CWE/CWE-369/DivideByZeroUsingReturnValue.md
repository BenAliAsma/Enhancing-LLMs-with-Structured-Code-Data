# Divide by zero using return value
Possible cases of division by zero when using the return value from functions.


## Example
The following example shows the use of a function with an error when using the return value and without an error.


```cpp

...
  a = getc(f);
  if (a < 123) ret = 123/a; // BAD
...
  if (a != 0) ret = 123/a; // GOOD
...

```

## References
* CERT Coding Standard: [INT33-C. Ensure that division and remainder operations do not result in divide-by-zero errors - SEI CERT C Coding Standard - Confluence](https://wiki.sei.cmu.edu/confluence/display/c/INT33-C.+Ensure+that+division+and+remainder+operations+do+not+result+in+divide-by-zero+errors).
* Common Weakness Enumeration: [CWE-369](https://cwe.mitre.org/data/definitions/369.html).
