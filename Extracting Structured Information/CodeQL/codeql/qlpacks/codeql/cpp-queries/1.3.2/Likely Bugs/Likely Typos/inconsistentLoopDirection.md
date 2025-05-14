# Inconsistent direction of for loop
A `for-loop` iteration expression goes backwards with respect of the initialization statement and condition expression.

This warning indicates that a `for-loop` might not function as intended.


## Recommendation
To fix this issue, check that the loop condition is correct and change the iteration expression to match.


## Example
In the following example, the initialization statement (`i = 0`) and the condition expression (`i < 100`) indicate that the intended iteration expression should have been incrementing, but instead a postfix decrement operator is used (`i--`).


```c
void f()
{
    for (signed char i = 0; i < 100; i--)
    {
        // code ...
    }
}
```
To fix this issue, change the iteration expression to match the direction of the initialization statement and the condition expression: `i++`.


## References
* [warning C6293: Ill-defined for-loop: counts down from minimum](https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2012/58teb7hd(v=vs.110))
* Common Weakness Enumeration: [CWE-835](https://cwe.mitre.org/data/definitions/835.html).
