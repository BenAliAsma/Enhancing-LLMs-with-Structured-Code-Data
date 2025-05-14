# Wrong NaN comparison
The special floating-point number `NaN` is defined to be different from all other floating-point numbers, including itself, when compared using the equality operators, `==` and `!=`.


## Recommendation
To check whether a variable `x` is `NaN` use the method `isNaN` that is defined on both `java.lang.Float` and `java.lang.Double`.


## Example
The expression `x == Double.NaN` is always false. This expression should be replaced by `Double.isNaN(x)`, which accurately identifies whether `x` is equal to `Double.NaN`.


## References
* Java Language Specification: [Numerical Equality Operators == and !=](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.21.1).
