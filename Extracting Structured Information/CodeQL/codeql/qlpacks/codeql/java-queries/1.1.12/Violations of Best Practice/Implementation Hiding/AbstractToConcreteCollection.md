# Cast from abstract to concrete collection
Most collections in the Java standard library are defined by an abstract interface (for example `java.util.List` or `java.util.Set`), which is implemented by a range of concrete classes and a range of wrappers. Normally, except when constructing an object, it is better to use the abstract types because this avoids assumptions about what the implementation is.

A cast from an abstract to a concrete collection makes the code brittle by ensuring it works only for one possible implementation class and not others. Usually, such casts are either an indication of over-reliance on concrete implementation types, or of the fact that the wrong abstract type was used.


## Recommendation
It is usually best to use the abstract type consistently in variable, field and parameter declarations.

There may be individual exceptions. For example, it is common to declare variables as `LinkedHashSet` rather than `Set` when the iteration order matters and only the `LinkedHashSet` implementation provides the right behavior.


## Example
The following example illustrates a situation where the wrong abstract type is used. The `List` interface does not provide a `poll` method, so the original code casts `queue` down to the concrete type `LinkedList`, which does. To avoid this downcasting, simply use the correct abstract type for this method, namely `Queue`. This documents the intent of the programmer and allows for various implementations of queues to be used by clients of this method.


```java
Customer getNext(List<Customer> queue) {
	if (queue == null)
		return null;
	LinkedList<Customer> myQueue = (LinkedList<Customer>)queue;  // AVOID: Cast to concrete type.
	return myQueue.poll();
}

Customer getNext(Queue<Customer> queue) {
	if (queue == null)
		return null;
	return queue.poll();  // GOOD: Use abstract type.
}

```

## References
* J. Bloch, *Effective Java (second edition)*, Item 52. Addison-Wesley, 2008.
* Java API Specification: [Collection](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collection.html).
* Common Weakness Enumeration: [CWE-485](https://cwe.mitre.org/data/definitions/485.html).
