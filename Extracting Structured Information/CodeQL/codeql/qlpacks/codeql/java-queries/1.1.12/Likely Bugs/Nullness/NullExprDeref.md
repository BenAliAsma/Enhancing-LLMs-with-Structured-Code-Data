# Dereferenced expression may be null
Dereferencing a `null` value leads to a `NullPointerException`.

An expression may be implicitly dereferenced if its type is a boxed primitive type, and it occurs in a context in which implicit unboxing occurs.


## Recommendation
Ensure that the expression does not have a `null` value when it is dereferenced. Use boxed types as appropriate to hold values that are potentially `null`.


## Example
In the following example implicit unboxing can cause a `NullPointerException` if `helper` is `null`.


```java
public int getID() {
    return helper == null ? null : helper.getID();
}

```
If the method is intended to return `null`, the return type should be changed to `Integer`.


## References
* The Java Tutorials: [Autoboxing and Unboxing](https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
