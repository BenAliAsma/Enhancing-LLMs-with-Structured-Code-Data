# Dangerous use of transformation after operation.
Search for places where the result of the multiplication is subjected to explicit conversion, not the arguments. Therefore, during the multiplication period, you can lose meaningful data.


## Example
The following example demonstrates erroneous and fixed methods for working with type conversion.


```cpp
...
  vUnsignedLong = (unsigned long)(vUnsignedInt*vUnsignedInt); // BAD
...
  vUnsignedLong = ((unsigned long)vUnsignedInt*vUnsignedInt); // GOOD
...

```

## References
* CERT C Coding Standard: [INT30-C. Ensure that unsigned integer operations do not wrap](https://wiki.sei.cmu.edu/confluence/display/c/INT30-C.+Ensure+that+unsigned+integer+operations+do+not+wrap).
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
