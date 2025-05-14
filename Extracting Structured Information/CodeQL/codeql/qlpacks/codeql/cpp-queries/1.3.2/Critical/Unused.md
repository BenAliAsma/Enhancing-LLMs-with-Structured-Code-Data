# Variable is assigned a value that is never read
This query finds variables that are assigned a value but are never read. This is usually an indication of a variable that has been orphaned due to changes in code, or a defect in the code due to the omission of the unused variable. The unused variables should be removed to avoid misuse.


## Recommendation
Examine the code to see if the variable is no longer needed. If it is unnecessary, remove the variable. Otherwise, update the relevant code to use the variable.


## Example

```cpp
{
	int foo = 1;
	... //foo is unused
}

```

## References
* SEI CERT C Coding Standard: [MSC13-C. Detect and remove unused values](https://wiki.sei.cmu.edu/confluence/display/c/MSC13-C.+Detect+and+remove+unused+values).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
