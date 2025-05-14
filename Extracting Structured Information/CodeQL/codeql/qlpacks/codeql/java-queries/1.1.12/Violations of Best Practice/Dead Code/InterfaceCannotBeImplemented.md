# Interface cannot be implemented
An interface that contains methods whose return types clash with protected methods on `java.lang.Object` can never be implemented, because methods cannot be overloaded based simply on their return type.


## Recommendation
If the interface is useful, name methods so that they do not clash with methods in `Object`. Otherwise you should delete the interface.


## Example
In the following example, the interface `I` is useless because the `clone` method must return type `java.lang.Object`:


```java
interface I {
    int clone();
}

class C implements I {
    public int clone() {
        return 23;
    }
}
```
Any attempt to implement the interface produces an error:

```

InterfaceCannotBeImplemented.java:6: clone() in C cannot override
  clone() in java.lang.Object; attempting to use incompatible return
  type
found   : int
required: java.lang.Object
  public int clone() {
             ^
1 error

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [9.2 Interface Members](https://docs.oracle.com/javase/specs/jls/se11/html/jls-9.html#jls-9.2).
