# Missing Override annotation
Java enables you to annotate methods that are intended to override a method in a superclass. Compilers are required to generate an error if such an annotated method does not override a method in a superclass, which provides increased protection from potential defects. An annotated method also improves code readability.


## Recommendation
Add an `@Override` annotation to a method that is intended to override a method in a superclass.


## Example
In the following example, `Triangle.getArea` overrides `Rectangle.getArea`, so it is annotated with `@Override`.


```java
class Rectangle
{
    private int w = 10, h = 10;
    public int getArea() { 
        return w * h; 
    }
}
 
class Triangle extends Rectangle
{
    @Override  // Annotation of an overriding method 
    public int getArea() { 
        return super.getArea() / 2; 
    }
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 36. Addison-Wesley, 2008.
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java API Specification: [Annotation Type Override](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Override.html).
* The Java Tutorials: [Predefined Annotation Types](https://docs.oracle.com/javase/tutorial/java/annotations/predefined.html).
