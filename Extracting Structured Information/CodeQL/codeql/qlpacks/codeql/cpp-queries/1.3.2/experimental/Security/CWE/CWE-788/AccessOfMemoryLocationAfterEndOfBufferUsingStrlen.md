# Access Of Memory Location After End Of Buffer
Potentially dangerous use of the strlen function to calculate the length of a string. The expression `buffer[strlen(buffer)] = 0` is potentially dangerous, if the variable buffer does not have a terminal zero, then access beyond the bounds of the allocated memory is possible, which will lead to undefined behavior. If terminal zero is present, then the specified expression is meaningless.

False positives include heavily nested strlen. This situation is unlikely.


## Recommendation
We recommend using another method for calculating the string length


## Example
The following example demonstrates an erroneous and corrected use of the strlen function.


```c
// BAD: if buffer does not have a terminal zero, then access outside the allocated memory is possible.

buffer[strlen(buffer)] = 0;


// GOOD: we will eliminate dangerous behavior if we use a different method of calculating the length. 
size_t len;
...
buffer[len] = 0

```

## References
* CERT C Coding Standard: [STR32-C. Do not pass a non-null-terminated character sequence to a library function that expects a string](https://wiki.sei.cmu.edu/confluence/display/c/STR32-C.+Do+not+pass+a+non-null-terminated+character+sequence+to+a+library+function+that+expects+a+string).
* Common Weakness Enumeration: [CWE-788](https://cwe.mitre.org/data/definitions/788.html).
