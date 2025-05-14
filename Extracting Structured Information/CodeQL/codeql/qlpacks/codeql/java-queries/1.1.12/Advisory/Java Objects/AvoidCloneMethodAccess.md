# Use of clone() method
Copying an object using the `Cloneable` interface and the `Object.clone` method is error-prone. This is because the `Cloneable` interface and the `clone` method are unusual:

* The `Cloneable` interface has no methods. Its only use is to trigger different behavior of `Object.clone`.
* `Object.clone` is protected.
* `Object.clone` creates a shallow copy without calling a constructor.
The first two points mean that a programmer must do two things to get a useful implementation of `clone`: first, make the class implement `Cloneable` to change the behavior of `Object.clone` so that it makes a copy instead of throwing a `CloneNotSupportedException`; second, override `clone` to make it public, to allow it to be called. Another consequence of `Cloneable` not having any methods is that it does not say anything about an object that implements it, which means that you cannot perform a polymorphic clone operation.

The third point, `Object.clone` creating a shallow copy, is the most serious one. A shallow copy shares internal state with the original object. This includes private fields that the programmer might not be aware of. A change to the internal state of the original object could affect the copy, and conversely the opposite is true, which could easily lead to unexpected behavior.


## Recommendation
Define either a dedicated copy method or a copy constructor (with a parameter whose type is the same as the type that declares the constructor). In most cases, this is at least as good as using the `Cloneable` interface and the `Object.clone` method, without the subtlety involved in implementing and using `clone` correctly.


## Example
In the following example, class `Galaxy` includes a copy constructor. Its parameter is of type `Galaxy`.


```java
public final class Galaxy {

    // This is the original constructor.
    public Galaxy (double aMass, String aName) {
        fMass = aMass;
        fName = aName;
    }

    // This is the copy constructor.
    public Galaxy(Galaxy aGalaxy) {
        this(aGalaxy.getMass(), aGalaxy.getName());
    }

    // ...
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 11. Addison-Wesley, 2008.
* Java API Specification: [Interface Cloneable](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Cloneable.html), [Object.clone](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#clone()).
