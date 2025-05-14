# Missing Javadoc for public type
A public class or interface that does not have a Javadoc comment makes an API more difficult to understand and maintain.


## Recommendation
Public classes and interfaces should be documented to make an API usable. For the purpose of code maintainability, it is also advisable to document non-public classes and interfaces.

Documentation for users of an API should be written using the standard Javadoc format. This can be accessed conveniently by users of an API from within standard IDEs, and can be transformed automatically into HTML format.


## Example
The following example shows a good Javadoc comment, which clearly explains what the class does, its author, and version.


```java
/**
 * The Stack class represents a last-in-first-out stack of objects. 
 *
 * @author  Joseph Bergin
 * @version 1.0, May 2000
 * Note that this version is not thread safe. 
 */
public class Stack {
// ...
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 44. Addison-Wesley, 2008.
* Help - Eclipse Platform: [Java Compiler Javadoc Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-javadoc.htm).
* Java SE Documentation: [How to Write Doc Comments for the Javadoc Tool](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html), [Requirements for Writing Java API Specifications](https://www.oracle.com/java/technologies/javase/api-specifications.html).
