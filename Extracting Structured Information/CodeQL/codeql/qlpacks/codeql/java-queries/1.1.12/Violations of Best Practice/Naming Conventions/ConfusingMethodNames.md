# Confusing method names because of capitalization
It is bad practice to have methods in a class with names that differ only in their capitalization. This can be confusing and lead to mistakes.


## Recommendation
Name the methods to make the distinction between them clear.


## Example
The following example shows a class that contains two methods: `toUri` and `toURI`. One or both of them should be renamed.


```java
public class InternetResource
{
	private String protocol;
	private String host;
	private String path;

	// ...

	public String toUri() {
		return protocol + "://" + host + "/" + path;
	}

	// ...

	public String toURI() {
		return toUri();
	}
}
```

## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, 17.N4. Prentice Hall, 2008.
