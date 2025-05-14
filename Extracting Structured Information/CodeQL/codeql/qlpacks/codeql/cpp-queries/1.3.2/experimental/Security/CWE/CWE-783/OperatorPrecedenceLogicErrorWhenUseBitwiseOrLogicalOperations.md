# Operator Precedence Logic Error When Use Bitwise Or Logical Operations
Find places of confusing use of logical and bitwise operations.


## Recommendation
We recommend using parentheses to explicitly emphasize priority.


## Example
The following example demonstrates fallacious and fixed methods of using logical and bitwise operations.


```c
bool a=1,b=0,c=1,res;
...
res = a||b^c; // BAD: possible priority error `res==1`
...
res = a||(b^c); // GOOD: `res==1`
... 
res = (a||b)^c; // GOOD: `res==0`

```

## References
* CERT C Coding Standard: [EXP00-C. Use parentheses for precedence of operation](https://wiki.sei.cmu.edu/confluence/display/c/EXP00-C.+Use+parentheses+for+precedence+of+operation).
* Common Weakness Enumeration: [CWE-783](https://cwe.mitre.org/data/definitions/783.html).
* Common Weakness Enumeration: [CWE-480](https://cwe.mitre.org/data/definitions/480.html).
