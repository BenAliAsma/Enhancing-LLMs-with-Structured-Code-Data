# Overloaded equals
Classes that define an `equals` method whose parameter type is not `Object` *overload* the `Object.equals` method instead of *overriding* it. This may not be intended.


## Recommendation
To *override* the `Object.equals` method, the parameter of the `equals` method must have type `Object`.


## Example
In the following example, the definition of class `BadPoint` does not override the `Object.equals` method. This means that `p.equals(q)` resolves to the default definition of `Object.equals` and returns `false`. Class `GoodPoint` correctly overrides `Object.equals`, so that `r.equals(s)` returns `true`.


```java
class BadPoint {
    int x;
    int y;

    BadPoint(int x, int y) {
        this.x = x;
        this.y = y;
    }

    // overloaded equals method -- should be avoided
    public boolean equals(BadPoint q) {
        return x == q.x && y == q.y;
    }
}

BadPoint p = new BadPoint(1, 2);
Object q = new BadPoint(1, 2);
boolean badEquals = p.equals(q); // evaluates to false

class GoodPoint {
    int x;
    int y;

    GoodPoint(int x, int y) {
        this.x = x;
        this.y = y;
    }

    // correctly overrides Object.equals(Object)
    public boolean equals(Object obj) {
        if (obj != null && getClass() == obj.getClass()) {
            GoodPoint q = (GoodPoint)obj;
            return x == q.x && y == q.y;
        }
        return false;
    }
}

GoodPoint r = new GoodPoint(1, 2);
Object s = new GoodPoint(1, 2);
boolean goodEquals = r.equals(s); // evaluates to true

```

## References
* J. Bloch, *Effective Java (second edition)*, Item 8. Addison-Wesley, 2008.
* Java Language Specification: [Overriding (by Instance Methods)](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.8.1), [Overloading](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.9).
* The Java Tutorials: [Overriding and Hiding Methods](https://docs.oracle.com/javase/tutorial/java/IandI/override.html).
