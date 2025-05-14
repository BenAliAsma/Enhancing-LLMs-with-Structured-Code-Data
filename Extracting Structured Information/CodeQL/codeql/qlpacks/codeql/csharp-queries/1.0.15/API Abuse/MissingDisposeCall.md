# Missing Dispose call
Classes that implement `IDisposable` and have members of `IDisposable` type should dispose those members in their `Dispose()` method.


## Recommendation
Add a call to `m.Dispose()` in your `Dispose` method for each member `m` that implements `IDisposable`.


## Example
In this example, the class `Bad` contains two disposable fields `stream1` and `stream2`, but only `stream1` is disposed in `Bad`'s `Dispose()` method.


```csharp
using System;
using System.IO;

class Bad : IDisposable
{
    private FileStream stream1 = new FileStream("a.txt", FileMode.Open);
    private FileStream stream2 = new FileStream("b.txt", FileMode.Open);

    public void Dispose()
    {
        stream1.Dispose();
    }
}

```
In the revised example, both `stream1` and `stream2` are disposed.


```csharp
using System;
using System.IO;

class Good : IDisposable
{
    private FileStream stream1 = new FileStream("a.txt", FileMode.Open);
    private FileStream stream2 = new FileStream("b.txt", FileMode.Open);

    public void Dispose()
    {
        stream1.Dispose();
        stream2.Dispose();
    }
}

```

## References
* MSDN: [IDisposable Interface](https://msdn.microsoft.com/en-us/library/system.idisposable.aspx).
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
* Common Weakness Enumeration: [CWE-459](https://cwe.mitre.org/data/definitions/459.html).
