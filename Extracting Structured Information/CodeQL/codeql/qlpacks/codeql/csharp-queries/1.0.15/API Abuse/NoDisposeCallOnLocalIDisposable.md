# Missing Dispose call on local IDisposable
Objects whose type implements `IDisposable` should be disposed of by calling `Dispose`.


## Recommendation
If possible, wrap the allocation of the object in a `using` block to automatically dispose of the object once the `using` block has completed.

If this is not possible, ensure that `Dispose` is called on the object. It is usually recommended to call `Dispose` within a `finally` block, to ensure that the object is disposed of even if an exception is thrown.


## Example
In this example, a `FileStream` is created, but it is not disposed of.


```csharp
using System;
using System.IO;

class Bad
{
    long GetLength(string file)
    {
        var stream = new FileStream(file, FileMode.Open);
        return stream.Length;
    }
}

```
In the revised example, a `using` statement is used to ensure that the file stream is properly closed.


```csharp
using System;
using System.IO;

class Good
{
    long GetLength(string file)
    {
        using (var stream = new FileStream(file, FileMode.Open))
            return stream.Length;
    }
}

```

## References
* MSDN: [IDisposable Interface](https://msdn.microsoft.com/en-us/library/system.idisposable.aspx).
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
* Common Weakness Enumeration: [CWE-459](https://cwe.mitre.org/data/definitions/459.html).
* Common Weakness Enumeration: [CWE-460](https://cwe.mitre.org/data/definitions/460.html).
