# Useless toString on String
There is no need to call `toString` on a `String` because it just returns the object itself. From the Java API Specification entry for `String.toString()`:

> `public String toString()`

This object (which is already a string!) is itself returned.


## Recommendation
Do not call `toString` on a `String` object.


## Example
The following example shows an unnecessary call to `toString` on the string `name`.


```java
public static void main(String args[]) {
	String name = "John Doe";
	
	// BAD: Unnecessary call to 'toString' on 'name'
	System.out.println("Hi, my name is " + name.toString());
	
	// GOOD: No call to 'toString' on 'name'
	System.out.println("Hi, my name is " + name);
}
```

## References
* Java API Specification: [String.toString()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#toString()).
