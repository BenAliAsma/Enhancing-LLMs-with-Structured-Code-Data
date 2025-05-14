# Typo in toString
A method named `tostring` may be a typographical error. `toString` (different capitalization) may have been intended instead.


## Recommendation
Ensure that any such method is intended to have this name. Even if it is, it may be better to rename it to avoid confusion with the inherited method `Object.toString`.


## Example
The following example shows a method named `tostring`. It may be better to rename it.


```java
public class Customer
{
	private String title;
	private String forename;
	private String surname;

	// ...

	public String tostring() {  // The method is named 'tostring'.
		return title + " " + forename + " " + surname;
	}
}
```

## References
* Java API Specification: [ Object.toString](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#toString()).
