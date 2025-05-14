# Assignment in return statement
An assignment is an expression. The value of an assignment expression is the value assigned to the variable. This can be useful, for example, when initializing two or more variables at once (for example, `a = b = 0;`). However, assigning to a local variable in the expression of a return statement is redundant because that value can never be read.


## Recommendation
Remove the redundant assignment from the `return` statement, leaving just the right-hand side of the assignment.


## Example
In the following example, consider the second assignment to `ret`. The variable goes out of scope when the method returns, and the value assigned to it is never read. Therefore, the assignment is redundant. Instead, the last line of the method can be changed to `return Math.max(ret, c);`


```java
public class Utilities
{
	public static int max(int a, int b, int c) {
		int ret = Math.max(a, b)
		return ret = Math.max(ret, c);  // Redundant assignment
	}
}
```

## References
* Java Language Specification: [ 14.17 The return Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.17), [ 15.26 Assignment Operators](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.26).
