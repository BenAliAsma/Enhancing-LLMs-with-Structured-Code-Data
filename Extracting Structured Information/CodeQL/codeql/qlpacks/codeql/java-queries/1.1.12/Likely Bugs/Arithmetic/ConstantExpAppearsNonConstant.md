# Expression always evaluates to the same value
Some expressions always evaluate to the same result, no matter what their subexpressions are:

* `x * 0` always evaluates to `0`.
* `x % 1` always evaluates to `0`.
* `x & 0` always evaluates to `0`.
* `x || true` always evaluates to `true`.
* `x && false` always evaluates to `false`.
Whenever `x` is not constant, such an expression is often a mistake.


## Recommendation
If the expression is supposed to evaluate to the same result every time it is executed, consider replacing the entire expression with its result.


## Example
The following method tries to determine whether `x` is even by checking whether `x % 1 == 0`.


```java
public boolean isEven(int x) {
	return x % 1 == 0; //Does not work
}

```
However, `x % 1 == 0` is always true when `x` is an integer. The correct check is `x % 2 == 0`.


```java
public boolean isEven(int x) {
    return x % 2 == 0; //Does work
}

```

## References
* Java Language Specification: [Multiplication Operator \*](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.17.1), [Remainder Operator %](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.17.3), [Integer Bitwise Operators &amp;, ^, and |](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.22.1), [Conditional-And Operator &amp;&amp;](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.23) and [Conditional-Or Operator ||](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.24).
