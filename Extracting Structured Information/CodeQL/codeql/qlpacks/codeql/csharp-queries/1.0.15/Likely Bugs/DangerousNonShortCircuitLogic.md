# Potentially dangerous use of non-short-circuit logic
The `|` and `&` logical operators, known as non-short circuit operators, should not be used. Using a non-short circuit operator reduces the efficiency of the program, is potentially confusing and can even lead to the program crashing if the first operand acts as a safety check for the second.


## Recommendation
If the non-short circuit operator is unintended then replace the operator with the short circuit equivalent. Sometime a non-short circuit operator is required because the operands have side effects. In this case it is more efficient to evaluate both operands separately and then use a short circuit operator to combine the results.


## Example
This example will crash because both parts of the conditional expression will be evaluated even if `a` is null.


```csharp
class DangerousNonShortCircuitLogic
{
    public static void Main(string[] args)
    {
        string a = null;
        if (a != null & a.ToLower() == "hello world")
        {
            Console.WriteLine("The string said hello world.");
        }
    }
}

```
The example is easily fixed by using the short circuit AND operator. The program produces no output but does not crash, unlike the previous example.


```csharp
class DangerousNonShortCircuitLogicFix
{
    public static void Main(string[] args)
    {
        string a = null;
        if (a != null && a.ToLower() == "hello world")
        {
            Console.WriteLine("The string said hello world.");
        }
    }
}

```

## References
* MSDN: [&amp; Operator](http://msdn.microsoft.com/en-us/library/sbf85k1c(v=vs.71).aspx)
* MSDN: [| Operator](http://msdn.microsoft.com/en-us/library/kxszd0kx(v=vs.71).aspx)
* Common Weakness Enumeration: [CWE-480](https://cwe.mitre.org/data/definitions/480.html).
* Common Weakness Enumeration: [CWE-691](https://cwe.mitre.org/data/definitions/691.html).
