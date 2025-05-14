# Typo in equals
A method named `equal` may be a typographical error. `equals` may have been intended instead.


## Recommendation
Ensure that any such method is intended to have this name. Even if it is, it may be better to rename it to avoid confusion with the inherited method `Object.equals`.


## Example
The following example shows a method named `equal`. It may be better to rename it.


```java
public class Complex
{
	private double real;
	private double complex;

	// ...

	public boolean equal(Object obj) {  // The method is named 'equal'.
		if (!getClass().equals(obj.getClass()))
			return false;
		Complex other = (Complex) obj;
		return real == other.real && complex == other.complex;
	}
}
```

## References
* Java API Specification: [ equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
