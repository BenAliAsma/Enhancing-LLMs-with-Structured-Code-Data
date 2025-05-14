# Potentially incorrect CompareTo(...) signature
If you wish to make a class comparable with type `T` then you should both make a `CompareTo(T)` method and inherit from the `IComparable<T>` interface. If a `CompareTo(T)` method has been added without the interface also being implemented, it is sometimes an indication that the programmer has forgotten to inherit the interface despite providing the implementation for it.


## Recommendation
The problem can be easily resolved by making the class implement `IComparable<T> ` (either directly or indirectly).


## Example
In this example, the developer has implemented a `CompareTo(Bad)` method, but has forgotten to add the corresponding `IComparable<Bad>` interface.


```csharp
using System;

class Bad
{
    public int CompareTo(Bad b) => 0;
}

```
In the revised example, the interface is added.


```csharp
using System;

class Good : IComparable<Good>
{
    public int CompareTo(Good g) => 0;
}

```

## References
* MSDN: [IComparable Interface](https://msdn.microsoft.com/en-us/library/system.icomparable.aspx)
