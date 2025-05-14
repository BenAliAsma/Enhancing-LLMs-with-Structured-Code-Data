# Dangerous non-short-circuit logic
Using a bitwise logical operator (`&` or `|`) on a boolean where a conditional-and or conditional-or operator (`&&` or `||`) is intended may yield the wrong result or cause an exception. This is especially true if the left-hand operand is a guard for the right-hand operand.

Typically, as in the example below, this kind of defect is introduced by simply mistyping the intended logical operator rather than any conceptual mistake by the programmer.


## Recommendation
If the right-hand side of an expression is only intended to be evaluated if the left-hand side evaluates to `true`, use a conditional-and.

Similarly, if the right-hand side of an expression is only intended to be evaluated if the left-hand side evaluates to `false`, use a conditional-or.


## Example
In the following example, the `hasForename` method is implemented correctly. For a forename to be valid it must be a non-null string with a non-zero length. The method has two expressions (`forename != null` and `forename.length() > 0`) to check these two properties. The second check is executed only if the first succeeds, because they are combined using a conditional-and operator (`&&`).

In contrast, although `hasSurname` looks almost the same, it contains a defect. Again there are two tests (`surname != null` and `surname.length() > 0`), but they are linked by a bitwise logical operator (`&`). Both sides of a bitwise logical operator are *always* evaluated, so if `surname` is `null` the `hasSurname` method throws a `NullPointerException`. To fix the defect, change `&` to `&&`.


```java
public class Person
{
	private String forename;
	private String surname;

	public boolean hasForename() {
		return forename != null && forename.length() > 0;  // GOOD: Conditional-and operator
	}

	public boolean hasSurname() {
		return surname != null & surname.length() > 0;  // BAD: Bitwise AND operator
	}

	// ...
}
```

## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 42. Addison-Wesley, 2005.
* Java Language Specification: [15.22.2 Boolean Logical Operators &amp;, ^, and |](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.22.2), [15.23 Conditional-And Operator &amp;&amp;](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.23), [15.24 Conditional-Or Operator ||](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.24).
* Common Weakness Enumeration: [CWE-691](https://cwe.mitre.org/data/definitions/691.html).
