# Recursive call to operator==
When writing an `operator==`, it is common to want to check whether or not an operand is null or check if both operands refer to the same object. Attempting to do this using `== ` will not work as it will simply call the overridden `operator==` that is being defined.


## Recommendation
There are two suggested ways to fix the problem. The tidiest approach is to simply replace the call to `operator==` with a call to `ReferenceEquals` (see code below). As a less preferable alternative, it is also possible to simply cast the parameters to `object `.


## Example
In this example the `==` overload will constantly call itself to compare the initial arguments passed to it with null.


```csharp
class Year
{
    private int year;
    private bool bc;

    public Year(int i)
    {
        year = Math.Abs(i);
        bc = i < 0;
    }

    public static bool operator ==(Year lhs, Year rhs)
    {
        // Both of these checks against null call operator== recursively.
        // They were probably intended to be checks for reference equality.
        if (lhs == null || rhs == null)
        {
            return false;
        }

        return lhs.year == rhs.year && lhs.bc == rhs.bc;
    }

    public static bool operator !=(Year lhs, Year rhs)
    {
        return !(lhs == rhs);
    }
}

```
The best fix is to use the `ReferenceEquals` method instead.


```csharp
class Year
{
    private int year;
    private bool bc;

    public Year(int i)
    {
        year = Math.Abs(i);
        bc = i < 0;
    }

    public static bool operator ==(Year lhs, Year rhs)
    {
        // Either of these alternative approaches works.
        if (ReferenceEquals(lhs, null) || (object)rhs == null)
        {
            return false;
        }

        return lhs.year == rhs.year && lhs.bc == rhs.bc;
    }

    public static bool operator !=(Year lhs, Year rhs)
    {
        return !(lhs == rhs);
    }
}

```

## References
* [Guidelines for Overloading Equals() and Operator== (C\#)](http://msdn.microsoft.com/en-us/library/ms173147%28VS.80%29.aspx)
