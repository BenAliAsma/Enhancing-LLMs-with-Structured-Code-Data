# Recursive call to Equals(object)
Recursive calls to `Equals(object)` are often made by accident whilst trying to forward to another `Equals` method with a more specific parameter type. This tends to result in a stack overflow exception.


## Recommendation
The solution is to check that the argument is of the appropriate type and then cast it when making the call.


## Example
In this example the `Equals(object)` method is called repeatedly when the programmer clearly meant to pass the object to the `Equals(C)` method but forgot to cast it in order to do so.


```csharp
class RecursiveEquals
{
    class C
    {
        private int i = 0;
        public override bool Equals(object rhs)
        {
            if (rhs.GetType() != this.GetType()) return false;
            return Equals(rhs);
        }

        public bool Equals(C rhs)
        {
            return (rhs != null && this.i == rhs.i);
        }
    }
}

```
This mistake is easily corrected by performing a cast before making the second call.


```csharp
class RecursiveEqualsFix
{
    class C
    {
        private int i = 0;
        public override bool Equals(object rhs)
        {
            if (rhs.GetType() != this.GetType()) return false;
            return Equals((C)rhs);
        }

        public bool Equals(C rhs)
        {
            return (rhs != null && this.i == rhs.i);
        }
    }
}

```
