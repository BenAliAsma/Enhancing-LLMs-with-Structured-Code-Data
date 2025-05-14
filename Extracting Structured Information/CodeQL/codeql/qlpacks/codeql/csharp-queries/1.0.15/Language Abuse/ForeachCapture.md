# Capturing a foreach variable
Avoid capturing the loop variable of `foreach` statements in lambda or delegate functions because the value captured varies depending on the version of C\# used. In C\# 5.0 and above the value captured is the *original* value but in earlier versions of C\# the value captured is the *current* value. If the function is used outside of the loop, then the function will behave differently in different versions of C\#.


## Recommendation
Make a copy of the loop variable, which can be safely captured in the lambda function without being changed by the loop.


## Example
The following example creates a list of functions, where each function is intended to print out a different number. However on older compilers, the actual output is `10 10 10 10 10 10 10 10 10 10`.


```csharp
var actions = new List<Fn>();
foreach (int i in Enumerable.Range(1, 10))
{
  // BAD: i is captured
  actions.Add(() => Console.Out.WriteLine(i));
}
actions.ForEach(fn => fn());

```
The bug can be fixed by introducing a new variable (`i2`) which does not capture the loop variable.


```csharp
var actions = new List<Fn>();
foreach (int i in Enumerable.Range(1, 10))
{
  int i2 = i; // GOOD: i is not captured
actions.Add(() => Console.Out.WriteLine(i2));
}
actions.ForEach(fn => fn());

```
The output of the new code is `1 2 3 4 5 6 7 8 9 10` as intended.


## References
* Eric Lippert's Blog: [Closing over the loop variable considered harmful](https://docs.microsoft.com/en-gb/archive/blogs/ericlippert/closing-over-the-loop-variable-considered-harmful).
* Common Weakness Enumeration: [CWE-758](https://cwe.mitre.org/data/definitions/758.html).
