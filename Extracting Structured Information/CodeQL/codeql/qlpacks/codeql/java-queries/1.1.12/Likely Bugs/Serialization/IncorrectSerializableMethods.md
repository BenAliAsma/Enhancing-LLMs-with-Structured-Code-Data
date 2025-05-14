# Serialization methods do not match required signature
A serializable object that defines its own serialization protocol using the methods `readObject`, `readObjectNoData` or `writeObject` must use the signature that is expected by the Java serialization framework. Otherwise, the default serialization mechanism is used.


## Recommendation
Make sure that the signatures of `readObject`, `readObjectNoData` and `writeObject` on serializable classes match these expected signatures:


```java
private void readObject(java.io.ObjectInputStream in)
     throws IOException, ClassNotFoundException;
private void readObjectNoData()
     throws ObjectStreamException;
private void writeObject(java.io.ObjectOutputStream out)
     throws IOException;
```

## Example
In the following example, `WrongNetRequest` defines `readObject`, `readObjectNoData` and `writeObject` using the wrong signatures. However, `NetRequest` defines them correctly.


```java
class WrongNetRequest implements Serializable {
	// BAD: Does not match the exact signature required for a custom 
	// deserialization protocol. Will not be called during deserialization.
	void readObject(ObjectInputStream in) {
		//...
	}
	
	// BAD: Does not match the exact signature required for a custom 
	// deserialization protocol. Will not be called during deserialization.
	void readObjectNoData() {
		//...
	}
	
	// BAD: Does not match the exact signature required for a custom 
	// serialization protocol. Will not be called during serialization.
	protected void writeObject(ObjectOutputStream out) {
		//...
	}
}

class NetRequest implements Serializable {
	// GOOD: Signature for a custom deserialization implementation.
	private void readObject(ObjectInputStream in) {
		//...
	}
	
	// GOOD: Signature for a custom deserialization implementation.
	private void readObjectNoData() {
		//...
	}
	
	// GOOD: Signature for a custom serialization implementation.
	private void writeObject(ObjectOutputStream out) {
		//...
	}
}
```

## References
* Java API Specification: [Serializable](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/Serializable.html).
* Oracle Technology Network: [Discover the secrets of the Java Serialization API](https://www.oracle.com/technical-resources/articles/java/serializationapi.html).
