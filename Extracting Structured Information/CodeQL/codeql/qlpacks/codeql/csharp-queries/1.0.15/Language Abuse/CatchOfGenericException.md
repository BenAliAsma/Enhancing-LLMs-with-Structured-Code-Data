# Generic catch clause
Catching all exceptions with a generic catch clause may be overly broad. This can make errors harder to diagnose when exceptions are caught unintentionally.


## Recommendation
If possible, catch only specific exception types to avoid catching unintended exceptions.


## Example
In the following example, a division by zero is incorrectly handled by catching all exceptions.


```csharp
double reciprocal(double input)
{
    try
    {
        return 1 / input;
    }
    catch
    {
        // division by zero, return 0
        return 0;
    }
}

```
In the corrected example, division by zero is correctly handled by only catching appropriate `DivideByZeroException` exceptions. Moreover, arithmetic overflow is now handled separately from division by zero by explicitly catching `OverflowException` exceptions.


```csharp
double reciprocal(double input)
{
    try
    {
        return 1 / input;
    }
    catch (DivideByZeroException)
    {
        return 0;
    }
    catch (OverflowException)
    {
        return double.MaxValue;
    }
}

```

## References
* Common Weakness Enumeration: [CWE-396](https://cwe.mitre.org/data/definitions/396.html).
