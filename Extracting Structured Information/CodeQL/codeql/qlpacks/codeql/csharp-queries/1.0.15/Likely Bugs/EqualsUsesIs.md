# Equals should not apply "is"
The method signature of `Equals` takes an object as an argument. It is therefore common to test whether the parameter is of the same type as the object that `Equals` is being called on. This should not be done using `is` because this technique does not limit the argument to being the exact same type as the object. It will not return false if the argument is a subtype of the object's type.

As an exception to the rule, it *is* acceptable to use `is` when your class is sealed, because the concern about subclassing does not then apply. That said, it is probably best to avoid it even then, because sealed classes can sometimes become unsealed later - it's sensible to try and avoid potential problems down the line.


## Recommendation
Call `GetType()` on the argument and compare it with the type of the current object instead.


## Example
The following example clearly demonstrates the problem with using `is`. The example outputs:

```
b does equal d.
d does not equal b.
```
This asymmetry violates the contract of the Equals method.


```csharp
class EqualsUsesIs
{
    class BaseClass
    {
        public override bool Equals(object obj)
        {
            return obj is BaseClass;
        }
    }

    class DClass : BaseClass
    {
        public override bool Equals(object obj)
        {
            return obj is DClass;
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
