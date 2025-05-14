# Empty interface
Empty interfaces are often used as a way of marking particular classes.


## Recommendation
In some languages, using a marker interface is a useful design pattern, but in C\# it is better to use custom attributes. Marker interfaces are always inherited and so they cannot be applied to a single class without applying it to all subclasses. Custom attributes do not have this limitation.


## Example
In this example, the `IsPrintable` interface has no defined methods and is simply being used as a marker.


```csharp
using System;

class Bad
{
    interface IsPrintable { }
    class Form1 : IsPrintable { }
}

```
The following example is better because it uses attributes instead.


```csharp
using System;

class Good
{
    [AttributeUsage(AttributeTargets.Class)]
    class PrintableAttribute : Attribute { }

    [Printable]
    class Form1 { }
}

```

## References
* Wikipedia: [Marker interface pattern](http://en.wikipedia.org/wiki/Marker_interface_pattern)
* Microsoft: [Using Attributes in C\#](https://docs.microsoft.com/en-us/dotnet/csharp/tutorials/attributes)
