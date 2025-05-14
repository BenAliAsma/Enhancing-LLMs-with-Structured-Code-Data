# Incorrect serialVersionUID field
A serializable class that uses the `serialVersionUID` field to act as an object version number must declare the field to be `final`, `static`, and of type `long` for it to be used by the Java serialization framework.


## Recommendation
Make sure that the `serialVersionUID` field in a serialized class is final, static, and of type `long`.


## Example
In the following example, `WrongNote` defines `serialVersionUID` using the wrong type, so that it is not used by the Java serialization framework. However, `Note` defines it correctly so that it is used by the framework.


```java
class WrongNote implements Serializable {
	// BAD: serialVersionUID must be static, final, and 'long'
	private static final int serialVersionUID = 1;
	
	//...
}

class Note implements Serializable {
	// GOOD: serialVersionUID is of the correct type
	private static final long serialVersionUID = 1L;
}
```

## References
* Java API Specification: [Serializable](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/Serializable.html).
* JavaWorld: [Ensure proper version control for serialized objects](https://www.infoworld.com/article/2071731/ensure-proper-version-control-for-serialized-objects.html).
