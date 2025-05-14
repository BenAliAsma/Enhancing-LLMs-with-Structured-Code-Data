# Serializable but no void constructor
A serializable class that is a subclass of a non-serializable class cannot be deserialized if its superclass does not declare a no-argument constructor. The Java serialization framework uses the no-argument constructor when it initializes the object instance that is created during deserialization. Deserialization fails with an `InvalidClassException` if its superclass does not declare a no-argument constructor.

The Java Development Kit API documentation states:

> To allow subtypes of non-serializable classes to be serialized, the subtype may assume responsibility for saving and restoring the state of the supertype's public, protected, and (if accessible) package fields. The subtype may assume this responsibility only if the class it extends has an accessible no-arg constructor to initialize the class's state. It is an error to declare a class `Serializable` if this is not the case. The error will be detected at runtime.


## Recommendation
Make sure that every non-serializable class that is extended by a serializable class has a no-argument constructor. Alternatively, consider defining a `writeReplace` method that replaces the `Serializable` class instance with a serialization proxy, so as to avoid direct deserialization of a class whose parent lacks a no-argument constructor.


## Example
In the following example, the class `WrongSubItem` cannot be deserialized because its superclass `WrongItem` does not declare a no-argument constructor. However, the class `SubItem` *can* be serialized because it declares a no-argument constructor.


```java
class WrongItem {
    private String name;

    // BAD: This class does not have a no-argument constructor, and throws an
    // 'InvalidClassException' at runtime.

    public WrongItem(String name) {
        this.name = name;
    }
}

class WrongSubItem extends WrongItem implements Serializable {
    public WrongSubItem() {
        super(null);
    }

    public WrongSubItem(String name) {
        super(name);
    }
}

class Item {
    private String name;

    // GOOD: This class declares a no-argument constructor, which allows serializable 
    // subclasses to be deserialized without error.
    public Item() {}

    public Item(String name) {
        this.name = name;
    }
}

class SubItem extends Item implements Serializable {
    public SubItem() { 
        super(null); 
    }

    public SubItem(String name) {
        super(name);
    }
}
```

## References
* Java API Specification: [Serializable](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/Serializable.html).
* J. Bloch, *Effective Java (second edition)*, Item 74. Addison-Wesley, 2008.
