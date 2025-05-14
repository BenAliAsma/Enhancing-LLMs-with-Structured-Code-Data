# Sign check of bitwise operation
Checking whether the result of a bitwise operation is greater than zero may yield unexpected results.


## Recommendation
It is more robust to check whether the result of the bitwise operation is *non-zero*.


## Example
In the following example, the expression assigned to variable `bad` is *not* a robust way to check that the `n`th bit of `x` is set. With the given values of `x` (all bits are set) and `n`, the expression `x & (1<<n)` has the value `-2147483648`, and the variable `bad` is assigned `false`, even though the 31st bit of `x` is, in fact, set.


```java
int x = -1;
int n = 31;

boolean bad = (x & (1<<n)) > 0;
```
In the following example, the expression assigned to variable `good` is a robust way to check that the `n`th bit of `x` is set. With the given values of `x` and `n`, the variable `good` is assigned `true`.


```java
int x = -1;
int n = 31;

boolean good = (x & (1<<n)) != 0;
```

## References
* Java Language Specification: [Integer Bitwise Operators &amp;, ^, and |](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.22.1).
