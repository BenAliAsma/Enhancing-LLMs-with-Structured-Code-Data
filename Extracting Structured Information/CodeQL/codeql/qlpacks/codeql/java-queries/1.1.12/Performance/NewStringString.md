# Inefficient String constructor
The `String` class is immutable, which means that there is no way to change the string that it represents. Consequently, there is rarely a need to copy a `String` object or construct a new instance based on an existing string, for example by writing something like `String hello = new String("hello")`. Furthermore, this practice is not memory efficient.


## Recommendation
The copied string is functionally indistinguishable from the argument that was passed into the `String` constructor, so you can simply omit the constructor call and use the argument passed into it directly. Unless an explicit copy of the argument string is needed, this is a safe transformation.


## Example
The following example shows three cases of copying a string using the `String` constructor, which is inefficient. In each case, simply removing the constructor call `new String` and leaving the argument results in better code and less memory churn.


```java
public void sayHello(String world) {
	// AVOID: Inefficient 'String' constructor
	String message = new String("hello ");

	// AVOID: Inefficient 'String' constructor
	message = new String(message + world);

	// AVOID: Inefficient 'String' constructor
	System.out.println(new String(message));
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 5. Addison-Wesley, 2008.
* Java API Specification: [String(String)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#%3Cinit%3E(java.lang.String)).
