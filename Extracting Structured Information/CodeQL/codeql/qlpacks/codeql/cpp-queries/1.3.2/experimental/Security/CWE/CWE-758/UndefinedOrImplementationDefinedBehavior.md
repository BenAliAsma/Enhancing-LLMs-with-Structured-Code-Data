# Errors Of Undefined Program Behavior
In some situations, the code constructs used may be executed in the wrong order in which the developer designed them. For example, if you call multiple functions as part of a single expression, and the functions have the ability to modify a shared resource, then the sequence in which the resource is changed can be unpredictable. These code snippets look suspicious and require the developer's attention.


## Recommendation
We recommend that you use more guaranteed, in terms of sequence of execution, coding techniques.


## Example
The following example demonstrates sections of code with insufficient execution sequence definition.


```c
intA = ++intA + 1; // BAD: undefined behavior when changing variable `intA`
...
intA++;
intA = intA + 1; // GOOD: correct design
...
char * buff;
...
if(funcAdd(buff)+fucDel(buff)>0) return 1; // BAD: undefined behavior when calling functions to change the `buff` variable
...
intA = funcAdd(buff);
intB = funcDel(buff);
if(intA+intB>0) return 1; // GOOD: correct design
```

## References
* CWE Common Weakness Enumeration: [ EXP10-C. Do not depend on the order of evaluation of subexpressions or the order in which side effects take place](https://wiki.sei.cmu.edu/confluence/display/c/EXP10-C.+Do+not+depend+on+the+order+of+evaluation+of+subexpressions+or+the+order+in+which+side+effects+take+place).
* Common Weakness Enumeration: [CWE-758](https://cwe.mitre.org/data/definitions/758.html).
