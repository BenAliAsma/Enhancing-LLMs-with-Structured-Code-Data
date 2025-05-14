# Annotation is extended or implemented
Although an annotation type is a special kind of interface that can be implemented by a concrete class, this is not its intended use. It is more likely that an annotation type should be used to annotate a class.


## Recommendation
Ensure that any annotations are used to annotate a class, unless they are really supposed to be extended or implemented by the class.


## Example
In the following example, the annotation `Deprecated` is implemented by the class `ImplementsAnnotation`.


```java
public abstract class ImplementsAnnotation implements Deprecated {
	// ...
}
```
The following example shows the intended use of annotations: to annotate the class `ImplementsAnnotationFix`.


```java
@Deprecated
public abstract class ImplementsAnnotationFix {
	// ...
}
```

## References
* Java Language Specification: [Annotation Types](https://docs.oracle.com/javase/specs/jls/se11/html/jls-9.html#jls-9.6).
* The Java Tutorials: [Annotations](https://docs.oracle.com/javase/tutorial/java/annotations/index.html).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
