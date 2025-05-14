# Transient field in non-serializable class
The `transient` modifier is used to identify fields that are not part of the persistent state of the class. As such, it only has an effect if the class is serializable, and has no purpose in a non-serializable class.

A field that is marked `transient` in a non-serializable class is likely to be a leftover from a time when the class was serializable.


## Recommendation
If the class is non-serializable, leave out the `transient` modifier. Otherwise, use the modifier, and ensure that the class (or a relevant supertype) implements `Serializable`.


## Example
The following example shows two fields that are declared `transient`. The modifier only has an effect in the class that implements `Serializable`.


```java
class State {
	// The 'transient' modifier has no effect here because
	// the 'State' class does not implement 'Serializable'.
	private transient int[] stateData;
}

class PersistentState implements Serializable {
	private int[] stateData;
	// The 'transient' modifier indicates that this field is not part of
	// the persistent state and should therefore not be serialized.
	private transient int[] cachedComputedData;
}
```

## References
* Java Language Specification: [8.3.1.3 transient Fields](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.3.1.3).
* Java Object Serialization Specification: [1.5 Defining Serializable Fields for a Class](https://docs.oracle.com/en/java/javase/11/docs/specs/serialization/serial-arch.html#defining-serializable-fields-for-a-class).
