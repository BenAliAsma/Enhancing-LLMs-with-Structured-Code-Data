# Magic numbers: use defined constant
This rule finds numbers that are used where a named constant already exists instead.


## Recommendation
Consider replacing the number with the existing named constant.


## Example

```csharp
class Circle
{
    private const double Pi = 3.14;
    private double radius;
    public double area()
    {
        return Math.Pow(radius, 2) * 3.14; // BAD: use the "Pi" constant
    }
    public double circumference()
    {
        return radius * 2 * 3.14; // BAD: use the "Pi" constant
    }
}

```

## References
* Robert C. Martin, *Clean Code - A handbook of agile software craftsmanship*, p. 295. Prentice Hall, 2008.
