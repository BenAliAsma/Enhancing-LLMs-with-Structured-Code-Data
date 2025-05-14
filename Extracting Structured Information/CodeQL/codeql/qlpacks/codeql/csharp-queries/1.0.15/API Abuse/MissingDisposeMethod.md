# Missing Dispose method
Classes that implement `IDisposable` and have members of `IDisposable` type should also declare/override `Dispose()`.


## Recommendation
Override the `Dispose()` method.


## Example
In the following example, `Bad` extends the `IDisposable` class `BadBase`, but does not override `Dispose()`.


```csharp
using System;
using System.IO;

class BadBase : IDisposable
{
    private FileStream stream1 = new FileStream("a.txt", FileMode.Open);

    public virtual void Dispose()
    {
        stream1.Dispose();
    }
}

class Bad : BadBase
{
    private FileStream stream2 = new FileStream("b.txt", FileMode.Open);
}

```
In the revised example, `Good` overrides `Dispose()`.


```csharp
using System;
using System.IO;

class GoodBase : IDisposable
{
    private FileStream stream1 = new FileStream("a.txt", FileMode.Open);

    public virtual void Dispose()
    {
        stream1.Dispose();
    }
}

class Good : BadBase
{
    private FileStream stream2 = new FileStream("b.txt", FileMode.Open);

    public override void Dispose()
    {
        base.Dispose();
        stream2.Dispose();
    }
}

```

## References
* MSDN: [IDisposable Interface](https://msdn.microsoft.com/en-us/library/system.idisposable.aspx).
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
* Common Weakness Enumeration: [CWE-459](https://cwe.mitre.org/data/definitions/459.html).
