# Equals should not apply "as"
The method signature of `Equals` takes an object as an argument. It is therefore common to test whether the parameter is of the same type as the object that `Equals` is being called on. This should not be done using `as` and then checking for null because this technique does not limit the argument to being the exact same type as the object. It will not return null if the argument is a subtype of the object's type.

As an exception to the rule, it *is* acceptable to use `as` when your class is sealed, because the concern about subclassing does not then apply. That said, it is probably best to avoid it even then, because sealed classes can sometimes become unsealed later - it's sensible to try and avoid potential problems down the line.


## Recommendation
Call `GetType()` on the argument and compare it with the type of the current object instead.


## Example
The following example clearly demonstrates the problem with using `as`. The example outputs:

```
b does equal d.
d does not equal b.
```
This asymmetry violates the contract of the Equals method.


```csharp
class EqualsUsesAs
{
    class BaseClass
    {
        public override bool Equals(object obj)
        {
            BaseClass objBase = obj as BaseClass;
            return objD != null;
        }
    }

    class DClass : BaseClass
    {
        public override bool Equals(object obj)
        {
            DClass objD = obj as DClass;
            return objD != null;
        }
    }

    public static void Main(string[] args)
    {
        BaseClass b = new BaseClass();
        DClass d = new DClass();
        Console.WriteLine("b " + (b.Equals(d) ? "does" : "does not") + " equal d.");
        Console.WriteLine("d " + (d.Equals(b) ? "does" : "does not") + " equal b.");
    }
}

```

## References
* MSDN: [Object.Equals Method (Object)](http://msdn.microsoft.com/en-us/library/bsc2ak47(v=vs.80).aspx)
