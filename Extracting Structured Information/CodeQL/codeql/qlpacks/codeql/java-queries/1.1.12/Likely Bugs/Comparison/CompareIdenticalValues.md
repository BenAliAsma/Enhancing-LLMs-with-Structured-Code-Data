# Comparison of identical values
If two identical expressions are compared (that is, checked for equality or inequality), this is typically an indication of a mistake, because the Boolean value of the comparison is always the same. Often, it indicates that the wrong qualifier has been used on a field access.

An exception applies to inequality (`!=`) and equality (`==`) tests of a floating point variable with itself: the special floating point value `NaN` ("not-a-number") is the only value that is not considered to be equal to itself. Thus, the test `x != x` where `x` is a `float` or `double` variable is equivalent to checking whether `x` is `NaN`, and similarly for `x == x`.


## Recommendation
It is never good practice to compare a value with itself. If you require constant behavior, use the Boolean literals `true` and `false`, rather than encoding them obscurely as `1 == 1` or similar.

If an inequality test (using `!=`) of a floating point variable with itself is intentional, it should be replaced by `Double.isNaN(...)` or `Float.isNaN(...)` for readability. Similarly, if an equality test (using `==`) of a floating point variable with itself is intentional, it should be replaced by `!Double.isNaN(...)` or `!Float.isNaN(...)`.


## Example
In the example below, the original version of `Customer` compares `id` with `id`, which always returns `true`. The corrected version of `Customer` includes the missing qualifier `o` in the comparison of `id` with `o.id`.


```java
class Customer {
	...
	public boolean equals(Object o) {
		if (o == null) return false;
		if (Customer.class != o.getClass()) return false;
		Customer other = (Customer)o;
		if (!name.equals(o.name)) return false;
		if (id != id) return false;  // Comparison of identical values
		return true;
	}
}

class Customer {
	...
	public boolean equals(Object o) {
		if (o == null) return false;
		if (Customer.class != o.getClass()) return false;
		Customer other = (Customer)o;
		if (!name.equals(o.name)) return false;
		if (id != o.id) return false;  // Comparison corrected
		return true;
	}
}
```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [15.21.1. Numerical Equality Operators](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.21.1).
