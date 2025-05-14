# Late Check Of Function Argument
Checking the function argument after calling the function itself. This situation looks suspicious and requires the attention of the developer. It may be necessary to add validation before calling the function


## Recommendation
We recommend checking before calling the function.


## Example
The following example demonstrates an erroneous and fixed use of function argument validation.


```c
if(len<0) return 1;
memset(dest, source, len); // GOOD: variable `len` checked before call

...

memset(dest, source, len); // BAD: variable `len` checked after call
if(len<0) return 1;

```

## References
* CWE Common Weakness Enumeration: [ CWE-20: Improper Input Validation](https://cwe.mitre.org/data/definitions/20.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
