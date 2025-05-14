# Dereferenced variable is always null
If a variable is dereferenced, for example as the qualifier in a method call, and the variable has a `null` value on all possible execution paths leading to the dereferencing, the dereferencing is guaranteed to result in a `NullReferenceException`.


## Recommendation
Ensure that the variable does not have a `null` value when it is dereferenced.


## Example
In the following examples, the condition `s.Length > 0` is only executed if `s` is `null`.


```csharp
using System;

namespace NullAlways
{
    class Bad
    {
        void DoPrint(string s)
        {
            if (s != null || s.Length > 0)
                Console.WriteLine(s);
        }
    }
}

```
In the revised example, the condition is guarded correctly by using `&&` instead of `||`.


```csharp
using System;

namespace NullAlways
{
    class Good
    {
        void DoPrint(string s)
        {
            if (s != null && s.Length > 0)
                Console.WriteLine(s);
        }
    }
}

```

## References
* Microsoft, [NullReferenceException Class](https://docs.microsoft.com/en-us/dotnet/api/system.nullreferenceexception).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
