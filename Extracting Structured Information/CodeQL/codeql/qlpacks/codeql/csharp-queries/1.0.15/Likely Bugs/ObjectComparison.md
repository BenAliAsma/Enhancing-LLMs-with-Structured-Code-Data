# Reference equality test on System.Object
If one or both operands of a comparison have the compile type `System.Object` or interface type, a reference comparison will be performed. This is often not what is wanted, especially when the runtime type of one of the operands declares `operator==`.

It should be noted that an exception is made for comparisons to read-only constants - a reference comparison is more likely to be intentional in such cases.


## Recommendation
Ensure a reference comparison is really the correct comparison to perform. If it is not then choose a more appropriate comparison. If reference comparison is intended, use `ReferenceEquals` instead for clarity.


## Example
In this example the console will output "False" because a reference comparison is performed instead of using the `==` operator overloaded in `AlwaysEqual`.


```csharp
class ObjectComparison
{
    class AlwaysEqual
    {
        public static bool operator ==(AlwaysEqual a, AlwaysEqual b)
        {
            return true;
        }
        public static bool operator !=(AlwaysEqual a, AlwaysEqual b)
        {
            return false;
        }
    }
    public static void Main(string[] args)
    {
        object a = new AlwaysEqual();
        AlwaysEqual b = new AlwaysEqual();
        Console.WriteLine(a == b);
    }
}

```

## References
* MSDN: [== Operator](http://msdn.microsoft.com/en-us/library/53k8ybth.aspx)
* Common Weakness Enumeration: [CWE-595](https://cwe.mitre.org/data/definitions/595.html).
