# Dereferenced variable may be null
If a variable is dereferenced, for example as the qualifier in a method call, and the variable may have a `null` value on some execution paths leading to the dereferencing, the dereferencing may result in a `NullReferenceException`.


## Recommendation
Ensure that the variable does not have a `null` value when it is dereferenced.


## Example
In the following example, the method `DoPrint()` dereferences its parameter `o` unconditionally, resulting in a `NullReferenceException` via the call `DoPrint(null)`.


```csharp
using System;

class Bad
{
    void DoPrint(object o)
    {
        Console.WriteLine(o.ToString());
    }

    void M()
    {
        DoPrint("Hello");
        DoPrint(null);
    }
}

```
In the revised example, the method `DoPrint()` guards the dereferencing with a `null` check.


```csharp
using System;

class Good
{
    void DoPrint(object o)
    {
        if (o != null)
            Console.WriteLine(o.ToString());
    }

    void M()
    {
        DoPrint("Hello");
        DoPrint(null);
    }
}

```

## References
* Microsoft, [NullReferenceException Class](https://docs.microsoft.com/en-us/dotnet/api/system.nullreferenceexception).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
