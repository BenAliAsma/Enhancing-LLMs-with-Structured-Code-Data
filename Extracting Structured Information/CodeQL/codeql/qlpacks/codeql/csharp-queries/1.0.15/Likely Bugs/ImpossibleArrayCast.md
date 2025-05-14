# Impossible array cast
Some casts between array types are guaranteed to fail at runtime: the cast from Object\[\] to String\[\] will always fail, even if all the elements of the array are strings. Casts identified by this check either fail immediately, or (in the case of arrays with parameterized types) cause an InvalidCastException later on in the code.


## Recommendation
Change the array creation expression to construct an array object of the right type.


## Example

```csharp
class ImpossibleArrayCast
{
    static void Main(string[] args)
    {
        // This will result in an InvalidCastException.
        String[] strs = (String[])new Object[] { "hello", "world" };
    }
}

```
