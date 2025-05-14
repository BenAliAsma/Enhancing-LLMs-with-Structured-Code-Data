# Typo in hashCode
A method named `hashcode` may be a typographical error. `hashCode` (different capitalization) may have been intended instead.


## Recommendation
Ensure that any such method is intended to have this name. Even if it is, it may be better to rename it to avoid confusion with the inherited method `Object.hashCode`.


## Example
The following example shows a method named `hashcode`. It may be better to rename it.


```java
public class Person
{
	private String title;
	private String forename;
	private String surname;

	// ...

	public int hashcode() {  // The method is named 'hashcode'.
		int hash = 23 * title.hashCode();
		hash ^= 13 * forename.hashCode();
		return hash ^ surname.hashCode();
	}
}
```

## References
* Java API Specification: [ Object.hashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#hashCode()).
