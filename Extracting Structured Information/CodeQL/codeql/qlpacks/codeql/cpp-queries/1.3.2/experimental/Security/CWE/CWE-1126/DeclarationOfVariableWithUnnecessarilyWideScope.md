# Errors When Using Variable Declaration Inside Loop
Using variables with the same name is dangerous. However, such a situation inside the while loop can create an infinite loop exhausting resources. Requires the attention of developers.


## Recommendation
We recommend not to use local variables inside a loop if their names are the same as the variables in the condition of this loop.


## Example
The following example demonstrates an erroneous and corrected use of a local variable within a loop.


```c
while(intIndex > 2)
{
  ...
  intIndex--;
  ...
} // GOOD: correct cycle
...
while(intIndex > 2)
{
  ...
  int intIndex;
  intIndex--;
  ...
} // BAD: the variable used in the condition does not change.

```

## References
* CERT C Coding Standard: [DCL01-C. Do not reuse variable names in subscopes](https://wiki.sei.cmu.edu/confluence/display/c/DCL01-C.+Do+not+reuse+variable+names+in+subscopes).
* Common Weakness Enumeration: [CWE-1126](https://cwe.mitre.org/data/definitions/1126.html).
