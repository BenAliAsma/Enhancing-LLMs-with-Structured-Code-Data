# Equals method does not inspect argument type
An implementation of `equals` must be able to handle an argument of any type, to avoid failing casts. Therefore, the implementation should inspect the type of its argument to see if the argument can be safely cast to the class in which the `equals` method is declared.


## Recommendation
Usually, an implementation of `equals` should check the type of its argument using `instanceof`, following the general pattern below.


```java
class A {
    // ...
    public final boolean equals(Object obj) {
        if (!(obj instanceof A)) {
        	return false;
        }
        A a = (A)obj;
        // ...further checks...
    }
    // ...
}
```
Using `instanceof` in this way has the added benefit that it includes a guard against null pointer exceptions: if `obj` is `null`, the check fails and `false` is returned. Therefore, after the check, it is guaranteed that `obj` is not `null`, and its fields can be safely accessed.

Whenever you use `instanceof` to check the type of the argument, you should declare the `equals` method `final`, so that subclasses are unable to cause a violation of the symmetry requirement of the `equals` contract by further overriding `equals`.

If you want subclasses to redefine the notion of equality by overriding `equals`, use `getClass` instead of `instanceof` to check the type of the argument. However, note that the use of `getClass` prevents any equality relationship between instances of a class and its subclasses, even when no additional state is added in a subclass.


## References
* J. Bloch, *Effective Java (second edition)*, Item 8. Addison-Wesley, 2008.
* Java API Specification: [Object.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
* Java Language Specification: [Type Comparison Operator instanceof](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.20.2).
