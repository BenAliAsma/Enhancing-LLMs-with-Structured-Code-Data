# Unchecked cast in Equals method
Since `Equals` takes an `object`, it can be passed an argument that has an arbitrary dynamic type: in particular, there is no guarantee that it is legal to cast the argument to the type that declares the `Equals` method. Code that does such a cast without checking the dynamic type of the argument first is therefore unsafe, and can break if an argument of the wrong dynamic type is passed in.


## Recommendation
To fix the problem, test the dynamic type of the argument before performing such a cast.


## Example
This example demonstrates casting of `object` to `UncheckedCastInEquals`. This can cause problems because if the object is not an `UncheckedCastInEquals` the program crashes with an `InvalidCastException`.


```csharp
class UncheckedCastInEquals
{
    private int i = 23;

    public override bool Equals(object obj)
    {
        UncheckedCastInEquals rhs = (UncheckedCastInEquals)obj;

        return i == rhs.i;
    }
}

```
A simple fix is to confirm the cast will succeed before performing it.


```csharp
class UncheckedCastInEqualsFix
{
    private int i = 23;

    public override bool Equals(object obj)
    {
        if (obj.GetType() != this.GetType())
            return false;

        UncheckedCastInEqualsFix rhs = (UncheckedCastInEqualsFix)obj;

        return i == rhs.i;
    }
}

```
