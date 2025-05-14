# Operator Precedence Logic Error When Use Bool Type
Finding places of confusing use of boolean type. For example, a unary minus does not work before a boolean type and an increment always gives true.


## Recommendation
we recommend making the code simpler.


## Example
The following example demonstrates erroneous and fixed methods for using a boolean data type.


```c
if(len=funcReadData()==0) return 1; // BAD: variable `len` will not equal the value returned by function `funcReadData()`
...
if((len=funcReadData())==0) return 1; // GOOD: variable `len` equal the value returned by function `funcReadData()`
...
bool a=true;
a++;// BAD: variable `a` does not change its meaning
bool b;
b=-a;// BAD: variable `b` equal `true`
...
a=false;// GOOD: variable `a` equal `false`
b=!a;// GOOD: variable `b` equal `false`

```

## References
* CERT C Coding Standard: [EXP00-C. Use parentheses for precedence of operation](https://wiki.sei.cmu.edu/confluence/display/c/EXP00-C.+Use+parentheses+for+precedence+of+operation).
* Common Weakness Enumeration: [CWE-783](https://cwe.mitre.org/data/definitions/783.html).
* Common Weakness Enumeration: [CWE-480](https://cwe.mitre.org/data/definitions/480.html).
