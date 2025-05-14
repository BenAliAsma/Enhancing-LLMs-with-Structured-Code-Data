# Missing Javadoc for public method or constructor
A public method or constructor that does not have a Javadoc comment makes an API more difficult to understand and maintain.


## Recommendation
Public methods and constructors should be documented to make an API usable. For the purpose of code maintainability, it is also advisable to document non-public methods and constructors.

The Javadoc comment should describe *what* the method or constructor does rather than *how*, to allow for any potential implementation change that is invisible to users of an API. It should include the following:

* A description of any preconditions or postconditions
* Javadoc tag elements that describe any parameters, return value, and thrown exceptions
* Any other important aspects such as side-effects and thread safety
Documentation for users of an API should be written using the standard Javadoc format. This can be accessed conveniently by users of an API from within standard IDEs, and can be transformed automatically into HTML format.


## Example
The following example shows a good Javadoc comment, which clearly explains what the method does, its parameter, return value, and thrown exception.


```java
/**
 * Extracts the user's name from the input arguments.
 *
 * Precondition: 'args' should contain at least one element, the user's name.
 *
 * @param  args            the command-line arguments.
 * @return                 the user's name (the first command-line argument).
 * @throws NoNameException if 'args' contains no element.
 */
public static String getName(String[] args) throws NoNameException {
	if(args.length == 0) {
		throw new NoNameException();
	} else {
		return args[0];
	}
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 44. Addison-Wesley, 2008.
* Help - Eclipse Platform: [Java Compiler Javadoc Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-javadoc.htm).
* Java SE Documentation: [How to Write Doc Comments for the Javadoc Tool](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html), [Requirements for Writing Java API Specifications](https://www.oracle.com/java/technologies/javase/api-specifications.html).
