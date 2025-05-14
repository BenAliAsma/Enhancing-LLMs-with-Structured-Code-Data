# Confusing method names because of overriding
If a method that would override another method but does not because the name is capitalized differently, there are two possibilities:

* The programmer intends the method to override the other method, and the difference in capitalization is a typographical error.
* The programmer does not intend the method to override the other method, in which case the similarity of the names is very confusing.

## Recommendation
If overriding *is* intended, make the capitalization of the two methods the same.

If overriding is *not* intended, consider naming the methods to make the distinction between them clear.


## Example
In the following example, `toString` has been wrongly capitalized as `tostring`. This means that objects of type `Customer` do not print correctly.


```java
public class Customer
{
	private String title;
	private String forename;
	private String surname;

	// ...

	public String tostring() {  // Incorrect capitalization of 'toString'
		return title + " " + forename + " " + surname;
	}
}
```

## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, &sect;17.N4. Prentice Hall, 2008.
