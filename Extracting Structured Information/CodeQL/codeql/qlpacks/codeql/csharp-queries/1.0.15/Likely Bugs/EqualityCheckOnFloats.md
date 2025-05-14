# Equality check on floating point values
Directly comparing two floating point values is dangerous due to the imprecision encountered when storing and performing arithmetic on them.


## Recommendation
Floating point numbers should be treated as equal if the difference between their values is within a certain margin of error. The appropriate margin of error depends on the situation in which it is used.

As a cautionary note, floating-point comparison is a non-trivial topic, and our documentation here takes a pragmatic approach rather than trying to do it justice. You are strongly advised to consult the references for further information.


## Example
Although you might expect this example to output "True" it actually outputs "False" due to the imprecise way floating point arithmetic is performed.


```csharp
class EqualityCheckOnFloats
{
    public static void Main(string[] args)
    {
        Console.WriteLine((0.1 + 0.2) == 0.3);
    }
}

```
The class should be changed to perform a comparison with a tolerance value as in the following example.


```csharp
class EqualityCheckOnFloatsFix
{
    public static void Main(string[] args)
    {
        const double EPSILON = 0.001;
        Console.WriteLine(Math.Abs((0.1 + 0.2) - 0.3) < EPSILON);
    }
}

```

## References
* Oracle Numerical Computation Guide: [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).
* Bruce Dawson: [Comparing Floating Point Numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
