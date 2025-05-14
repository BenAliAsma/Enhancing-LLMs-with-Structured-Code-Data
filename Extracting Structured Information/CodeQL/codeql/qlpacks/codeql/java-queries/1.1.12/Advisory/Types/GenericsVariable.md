# Non-parameterized variable
The use of generics in Java improves compile-time type safety and code readability. Users of a class or interface that has been designed using generic types should therefore make use of parameterized instances in variable declarations, method return types, and constructor calls.


## Recommendation
Provide type parameters to generic classes and interfaces where possible.

Note that converting legacy code to use generics may have to be done carefully in order to preserve the existing functionality of an API; for detailed guidance, see the references.


## Example
The following example is poorly written because it uses raw types. This makes it more error prone because the compiler is less able to perform type checks.


```java
public List constructRawList(Object o) {
    List list;  // Raw variable declaration
    list = new ArrayList();  // Raw constructor call
    list.add(o);
    return list;  // Raw method return type (see signature above)
}
```
A parameterized version can be easily made and is much safer.


```java
public <T> List<T> constructParameterizedList(T o) {
    List<T> list;  // Parameterized variable declaration
    list = new ArrayList<T>();  // Parameterized constructor call
    list.add(o);
    return list;  // Parameterized method return type (see signature above)
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 23. Addison-Wesley, 2008.
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* The Java Tutorials: [Generics](https://docs.oracle.com/javase/tutorial/java/generics/), [Converting Legacy Code to Use Generics](https://docs.oracle.com/javase/tutorial/extra/generics/convert.html).
