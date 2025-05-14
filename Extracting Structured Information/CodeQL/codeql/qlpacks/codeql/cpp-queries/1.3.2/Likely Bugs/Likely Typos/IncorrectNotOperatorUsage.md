# Incorrect 'not' operator usage
This rule finds logical-not operator usage as an operator for in a bit-wise operation.

Due to the nature of logical operation result value, only the lowest bit could possibly be set, and it is unlikely to be intent in bitwise operations. Violations are often indicative of a typo, using a logical-not (`!`) operator instead of the bit-wise not (`~`) operator.

This rule is restricted to analyze bit-wise and (`&`) and bit-wise or (`|`) operation in order to provide better precision.

This rule ignores instances where a double negation (`!!`) is explicitly used as the operator of the bitwise operation, as this is a commonly used as a mechanism to normalize an integer value to either 1 or 0.

NOTE: It is not recommended to use this rule in kernel code or older C code as it will likely find several false positive instances.


## Recommendation
Carefully inspect the flagged expressions. Consider the intent in the code logic, and decide whether it is necessary to change the not operator.


## Example
Here is an example of this issue and how it can be fixed:


```cpp
#define FLAGS   0x4004

void f_warning(int i)
{
    // BAD: the usage of the logical not operator in this case is unlikely to be correct
    // as the output is being used as an operator for a bit-wise and operation
    if (i & !FLAGS)
    {
        // code
    }
}

void f_fixed(int i)
{
    if (i & ~FLAGS) // GOOD: Changing the logical not operator for the bit-wise not operator would fix this logic
    {
        // code
    }
}

```
In other cases, particularly when the expressions have `bool` type, the fix may instead be of the form `a && !b`.


## References
* [warning C6317: incorrect operator: logical-not (!) is not interchangeable with ones-complement (~)](https://docs.microsoft.com/en-us/visualstudio/code-quality/c6317?view=vs-2017)
* Common Weakness Enumeration: [CWE-480](https://cwe.mitre.org/data/definitions/480.html).
