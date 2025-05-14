# Find the wrong use of the umask function.
Finding for function calls that set file permissions that may have errors in use. Incorrect arithmetic for calculating the resolution mask, using the same mask in opposite functions, using a mask that is too wide.


## Example
The following example demonstrates erroneous and fixed ways to use functions.


```cpp
...
  umask(0); // BAD
...
  maskOut = S_IRWXG | S_IRWXO;
  umask(maskOut); // GOOD
  ...
  fchmod(fileno(fp), 0555 - maskOut); // BAD 
  ...
  fchmod(fileno(fp), 0555 & ~maskOut); // GOOD
...
  umask(0666);
  chmod(pathname, 0666); // BAD
...
  umask(0022);
  chmod(pathname, 0666); // GOOD
...

```

## References
* CERT C Coding Standard: [FIO06-C. Create files with appropriate access permissions](https://wiki.sei.cmu.edu/confluence/display/c/FIO06-C.+Create+files+with+appropriate+access+permissions).
* Common Weakness Enumeration: [CWE-266](https://cwe.mitre.org/data/definitions/266.html).
* Common Weakness Enumeration: [CWE-264](https://cwe.mitre.org/data/definitions/264.html).
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
* Common Weakness Enumeration: [CWE-560](https://cwe.mitre.org/data/definitions/560.html).
* Common Weakness Enumeration: [CWE-687](https://cwe.mitre.org/data/definitions/687.html).
