# Inconsistent Equals(object) and GetHashCode()
A class that overrides only one of `Equals(object)` and `GetHashCode()` is likely to violate the contract of the `GetHashCode()` method. The contract requires that `GetHashCode()` gives the same integer result for any two equal objects. Not enforcing this property may cause unexpected results when storing and retrieving objects of such a class in a hashing data structure.


## Recommendation
Provide an implementation of the missing method that is consistent with the present method.


## Example
In the following example, the class `Bad` overrides `Equals(object)` but not `GetHashCode()`.


```csharp
using System;

class Bad
{
    private int id;

    public Bad(int Id) { this.id = Id; }

    public override bool Equals(object other)
    {
        if (other is Bad b && b.GetType() == typeof(Bad))
            return this.id == b.id;
        return false;
    }
}

```
In the revised example, the class `Good` overrides both `Equals(object)` and `GetHashCode()`.


```csharp
using System;

class Good
{
    private int id;

    public Good(int Id) { this.id = Id; }

    public override bool Equals(object other)
    {
        if (other is Good b && b.GetType() == typeof(Good))
            return this.id == b.id;
        return false;
    }

    public override int GetHashCode() => id;
}

```

## References
* MSDN: [Object.Equals Method (Object)](https://msdn.microsoft.com/en-us/library/bsc2ak47.aspx), [Object.GetHashCode Method](https://msdn.microsoft.com/en-us/library/system.object.gethashcode.aspx).
* Common Weakness Enumeration: [CWE-581](https://cwe.mitre.org/data/definitions/581.html).
