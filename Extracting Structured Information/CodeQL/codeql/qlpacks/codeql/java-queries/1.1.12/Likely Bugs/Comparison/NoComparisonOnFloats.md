# Equality test on floating point values
Equality tests on floating point values may lead to unexpected results because of arithmetic imprecision. For example, the expression `23.42f==23.42` evaluates to `false`.


## Recommendation
Instead of testing for *exact equality* between floating point values, check that the difference between the values is within an appropriate error margin.

Alternatively, if you do not want any inaccuracy when testing for equality, use one of the following instead of floating point values:

* `BigDecimal` class. This can store decimal values with higher precision.
* `long` type. Because this is an integer type, you must convert any decimal values to whole values. For example, represent $1.43 as 143 cents.

## Example
In the following example, `(0.1 + 0.2) == 0.3` evaluates to `false`, even though you would expect it to evaluate to `true`. This is because of the imprecision of floating point data types.


```java
class NoComparisonOnFloats
{
    public static void main(String[] args)
    {
        System.out.println((0.1 + 0.2) == 0.3);
    }
}
```
In the following improved example, the test for equality is performed by calculating the difference between the two values, and checking if the difference is within the error margin, `EPSILON`.


```java
class NoComparisonOnFloats
{
    public static void main(String[] args)
    {
        final double EPSILON = 0.001;
        System.out.println(Math.abs((0.1 + 0.2) - 0.3) < EPSILON);
    }
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 48. Addison-Wesley, 2008.
* Numerical Computation Guide: [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).
