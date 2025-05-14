# Errors When Using Bit Operations
Using bitwise operations can be a mistake in some situations. For example, if parameters are evaluated in an expression and the function should be called only upon certain test results. These bitwise operations look suspicious and require developer attention.


## Recommendation
We recommend that you evaluate the correctness of using the specified bit operations.


## Example
The following example demonstrates the erroneous and fixed use of bit and logical operations.


```c
if(len>0 & memset(buf,0,len)) return 1; // BAD: `memset` will be called regardless of the value of the `len` variable. moreover, one cannot be sure that it will happen after verification
...
if(len>0 && memset(buf,0,len)) return 1; // GOOD: `memset` will be called after the `len` variable has been checked.
...

```

## References
* CWE Common Weakness Enumeration: [ CWE-691: Insufficient Control Flow Management](https://cwe.mitre.org/data/definitions/691.html).
* Common Weakness Enumeration: [CWE-691](https://cwe.mitre.org/data/definitions/691.html).
