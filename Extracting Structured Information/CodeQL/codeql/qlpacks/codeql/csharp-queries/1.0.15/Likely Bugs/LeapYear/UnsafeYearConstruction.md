# Unsafe year argument for 'DateTime' constructor
When creating a `System.DateTime` object by setting the year, month, and day in the constructor by performing an arithmetic operation on a different `DateTime` object, there is a risk that the date you are setting is invalid.

On a leap year, such code may throw an `ArgumentOutOfRangeException` with a message of `"Year, Month, and Day parameters describe an unrepresentable DateTime."`


## Recommendation
Creating a `System.DateTime` object based on a different `System.DateTime` object, use the appropriate methods to manipulate the date instead of arithmetic.


## Example
In this example, we are incrementing/decrementing the current date by one year when creating a new `System.DateTime` object. This may work most of the time, but on any given February 29th, the resulting value will be invalid.


```csharp
using System;
public class UnsafeYearConstructionBad
{
    public UnsafeYearConstructionBad()
    {
        DateTime Start;
        DateTime End;
        var now = DateTime.UtcNow;
        // the base-date +/- n years may not be a valid date.
        Start = new DateTime(now.Year - 1, now.Month, now.Day, 0, 0, 0, DateTimeKind.Utc);
        End = new DateTime(now.Year + 1, now.Month, now.Day, 0, 0, 1, DateTimeKind.Utc);
    }
}
```
To fix this bug, we add/substract years to the current date by calling `AddYears` method on it.


```csharp
using System;
public class UnsafeYearConstructionGood
{
    public UnsafeYearConstructionGood()
    {
        DateTime Start;
        DateTime End;
        var now = DateTime.UtcNow;
        Start = now.AddYears(-1).Date;
        End = now.AddYears(-1).Date.AddSeconds(1);
    }
}
```

## References
* [System.DateTime Struct](https://docs.microsoft.com/en-us/dotnet/api/system.datetime?view=netframework-4.8).
