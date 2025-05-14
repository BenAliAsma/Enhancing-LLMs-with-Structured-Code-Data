# Non-serializable comparator
A class that implements `java.util.Comparator` and is used to construct a sorted collection needs to be serializable. An ordered collection (such as a `java.util.TreeMap`) that is constructed using a comparator serializes successfully only if the comparator is serializable.

The `Collection`s in the Java Standard Library that require a comparator (`TreeSet`, `TreeMap`, `PriorityQueue`) all call `ObjectOutputStream.defaultWriteObject`, which tries to serialize every non-static, non-transient field in the class. As the comparator is stored in a field in these collections, the attempt to serialize a non-serializable comparator throws a `java.io.NotSerializableException`.


## Recommendation
Comparators should be serializable if they are used in sorted collections that may be serialized. In most cases, simply changing the comparator so it also implements `Serializable` is enough. Comparators that have internal state may require additional changes (for example, custom `writeObject` and `readObject` methods). In these cases, it is best to follow general best practices for serializable objects (see references below).


## Example
In the following example, `WrongComparator` is not serializable because it does not implement `Serializable`. However, `StringComparator` is serializable because it does implement `Serializable`.


```java
// BAD: This is not serializable, and throws a 'java.io.NotSerializableException'
// when used in a serializable sorted collection.
class WrongComparator implements Comparator<String> {
    public int compare(String o1, String o2) {
        return o1.compareTo(o2);
    }
}

// GOOD: This is serializable, and can be used in collections that are meant to be serialized.
class StringComparator implements Comparator<String>, Serializable {
    private static final long serialVersionUID = -5972458403679726498L;

    public int compare(String arg0, String arg1) {
        return arg0.compareTo(arg1);
    }
}
```

## References
* Java API Specification: [Comparator](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Comparator.html), [ObjectOutputStream](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/ObjectOutputStream.html), [Serializable](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/Serializable.html).
