# Hashed value without hashCode definition
Classes that define an `equals` method but no `hashCode` method can lead to unexpected results if instances of those classes are stored in a hashing data structure. Hashing data structures expect that hash codes fulfill the contract that two objects that `equals` considers equal should have the same hash code. This contract is likely to be violated by such classes.


## Recommendation
Every class that implements a custom `equals` method should also provide an implementation of `hashCode`.


## Example
In the following example, class `Point` has no implementation of `hashCode`. Calling `hashCode` on two distinct `Point` objects with the same coordinates would probably result in different hash codes. This would violate the contract of the `hashCode` method, in which case objects of type `Point` should not be stored in hashing data structures.


```java
class Point {
    int x;
    int y;

    Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public boolean equals(Object o) {
    	if (!(o instanceof Point)) return false;
    	Point q = (Point)o;
    	return x == q.x && y == q.y;
    }
}
```
In the modification of the above example, the implementation of `hashCode` for class `Point` is suitable because the hash code is computed from exactly the same fields that are considered in the `equals` method. Therefore, the contract of the `hashCode` method is fulfilled.


```java
class Point {
    int x;
    int y;

    Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public boolean equals(Object o) {
        if (!(o instanceof Point)) return false;
        Point q = (Point)o;
        return x == q.x && y == q.y;
    }

    // Implement hashCode so that equivalent points (with the same values of x and y) have the
    // same hash code
    public int hashCode() {
        int hash = 7;
        hash = 31*hash + x;
        hash = 31*hash + y;
        return hash;
    }
}

```

## References
* J. Bloch, *Effective Java (second edition)*, Item 9. Addison-Wesley, 2008.
* Java API Specification: [Object.equals](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)), [Object.hashCode](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#hashCode()).
* IBM developerWorks: [Java theory and practice: Hashing it out](https://www.ibm.com/developerworks/java/library/j-jtp05273/index.html).
