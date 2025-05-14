# Unnecessarily complex boolean expression
There are a number of boolean expression patterns that can easily be rewritten to make them simpler. Boolean expressions involving comparisons with boolean literals, ternary conditionals with a boolean literal as one of the results, double negations, or negated comparisons can all be changed to equivalent and simpler expressions.


## Recommendation
If `A` and `B` are expressions of boolean type, you can simplify them using the rewrites shown below. These rewrites are equally suitable for the primitive type `boolean` and the boxed type `Boolean`.

<table><tbody> <tr><th>Expression</th><th></th><th>Simplified expression</th><th></th></tr> <tr><td><code>A == true</code></td><td></td><td><code>A</code></td><td>(\*)</td></tr> <tr><td><code>A != false</code></td><td></td><td><code>A</code></td><td>(\*)</td></tr> <tr><td><code>A == false</code></td><td></td><td><code>!A</code></td><td></td></tr> <tr><td><code>A != true</code></td><td></td><td><code>!A</code></td><td></td></tr> <tr><td><code>A ? true : B</code></td><td></td><td><code>A || B</code></td><td></td></tr> <tr><td><code>A ? B : false</code></td><td></td><td><code>A &amp;&amp; B</code></td><td></td></tr> <tr><td><code>A ? B : true</code></td><td></td><td><code>!A || B</code></td><td></td></tr> <tr><td><code>A ? false : B</code></td><td></td><td><code>!A &amp;&amp; B</code></td><td></td></tr> <tr><td><code>A ? true : false</code></td><td></td><td><code>A</code></td><td>(\*)</td></tr> <tr><td><code>A ? false : true</code></td><td></td><td><code>!A</code></td><td></td></tr> <tr><td><code>!!A</code></td><td></td><td><code>A</code></td><td>(\*)</td></tr> </tbody></table>
Note that all of these rewrites yield completely equivalent code, except possibly for those marked with (\*) when `A` has type `Boolean`. These can depend on the context if `null` values are involved. This is because a comparison or test of a `Boolean` will perform an automatic unboxing conversion that throws a `NullPointerException` if `null` is encountered, so the rewrites marked (\*) are only completely equivalent if the surrounding context of the expression unboxes the value, for example by occurring as the condition in an `if` statement.

In addition to the rewrites above, negated comparisons can also be simplified in the following way:

<table><tbody> <tr><th>Expression</th><th></th><th>Simplified expression</th></tr> <tr><td><code>!(A == B)</code></td><td></td><td><code>A != B</code></td></tr> <tr><td><code>!(A != B)</code></td><td></td><td><code>A == B</code></td></tr> <tr><td><code>!(A &lt; B)</code></td><td></td><td><code>A &gt;= B</code></td></tr> <tr><td><code>!(A &gt; B)</code></td><td></td><td><code>A &lt;= B</code></td></tr> <tr><td><code>!(A &lt;= B)</code></td><td></td><td><code>A &gt; B</code></td></tr> <tr><td><code>!(A &gt;= B)</code></td><td></td><td><code>A &lt; B</code></td></tr> </tbody></table>

## Example
In the following example the method `f2` is a straightforward simplification of `f1`; they will both throw a `NullPointerException` if a `null` is encountered. A similar rewrite of `g1` would however result in a method with a different meaning, and a rewrite along the lines of `g2` might be more appropriate. In any case, care should be taken to ensure correct behavior for `null` values when the boxed type `Boolean` is used.


```java
boolean f1(List<Boolean> l) {
	for(Boolean x : l) {
		if (x == true) return true;
	}
	return false;
}

boolean f2(List<Boolean> l) {
	for(Boolean x : l) {
		if (x) return true;
	}
	return false;
}

void g1(List<Boolean> l1) {
	List<Boolean> l2 = new ArrayList<Boolean>();
	for(Boolean x : l1) {
		l2.add(x == true);
	}
}

void g2(List<Boolean> l1) {
	List<Boolean> l2 = new ArrayList<Boolean>();
	for(Boolean x : l1) {
		if (x == null) {
			// handle null case
		}
		l2.add(x);
	}
}

```

## References
* Java Language Specification: [Logical Complement Operator !](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.15.6), [Boolean Equality Operators == and !=](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.21.2), [Conditional-And Operator &amp;&amp;](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.23), [Conditional-Or Operator ||](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.24), [Conditional Operator ? :](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.25).
