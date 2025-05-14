# Type mismatch in conditional expression
Conditional expressions of the form `(p ? e1 : e2)` can yield unexpected results if `e1` and `e2` have distinct primitive types.


## Example
The following example illustrates the most confusing case, which occurs when one branch has type `char` and the other branch does not have type `char`.


```java
int i = 0;
System.out.print(true ? 'x' : 0); // prints "x"
System.out.print(true ? 'x' : i); // prints "120"
```
This unexpected result is due to binary numeric promotion of `'x'` from `char` to `int`. For details on the result type of the conditional operator, see the references.


## Recommendation
When using the ternary conditional operator with numeric operands, the second and third operand should have the same numeric type. This avoids potentially unexpected results caused by binary numeric promotion.


## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 8. Addison-Wesley, 2005.
* Java Language Specification: [Conditional Operator ? :](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.25).
