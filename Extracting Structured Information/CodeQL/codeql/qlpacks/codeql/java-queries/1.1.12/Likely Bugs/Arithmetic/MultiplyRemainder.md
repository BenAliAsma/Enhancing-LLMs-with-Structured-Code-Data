# Multiplication of remainder
Using the remainder operator `%` with the multiplication operator may not give you the result that you expect unless you use parentheses. This is because the remainder operator has the same precedence as the multiplication operator, and the operators are left-associative.


## Recommendation
When you use the remainder operator with the multiplication operator, ensure that the expression is evaluated as you expect. If necessary, add parentheses.


## Example
Consider a time in milliseconds, represented by `t`. To calculate the number of milliseconds remaining after the time has been converted to whole minutes, you might write `t % 60 * 1000`. However, this is equal to `(t % 60) * 1000`, which gives the wrong result. Instead, the expression should be `t % (60 * 1000)`.


## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 35. Addison-Wesley, 2005.
* The Java Tutorials: [Operators](https://docs.oracle.com/javase/tutorial/java/nutsandbolts/operators.html).
