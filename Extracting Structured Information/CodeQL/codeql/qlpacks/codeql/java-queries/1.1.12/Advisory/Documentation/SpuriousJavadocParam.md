# Spurious Javadoc @param tags
Javadoc comments for public methods, constructors and generic classes should use the `@param` tag to describe the available parameters and type parameters. If the comment includes any empty, incorrect or outdated parameter names then this will make the documentation more difficult to read.


## Recommendation
The Javadoc comment for a method, constructor or generic class should always use non-empty `@param` values that match actual parameter or type parameter names.


## Example
The following example shows good and bad Javadoc comments that use the `@param` tag.


```java
/**
 * BAD: The following param tag is empty.
 *
 * @param   
 */ 
public void emptyParamTag(int p){ ... }


/**
 * BAD: The following param tag has a misspelled value.
 *
 * @param prameter The parameter's value.
 */ 
public void typo(int parameter){ ... }


/**
 * BAD: The following param tag appears to be outdated
 * since the method does not take any parameters.
 *
 * @param sign The number's sign.
 */ 
public void outdated(){ ... }


/**
 * BAD: The following param tag uses html within the tag value.
 *
 * @param <code>ordinate</code> The value of the y coordinate.
 */ 
public void html(int ordinate){ ... }


/**
 * BAD: Invalid syntax for type parameter.
 *
 * @param T The type of the parameter.
 * @param parameter The parameter value.
 */ 
public <T> void parameterized(T parameter){ ... }

/**
 * BAD: The following param tag refers to a non-existent type parameter.
 * 
 * @param <X> The type of the elements.
 */
class Generic<T> { ... }

/**
 * GOOD: A proper Javadoc comment.
 *
 * This method calculates the absolute value of a given number.
 *
 * @param <T> The number's type.
 * @param x The number to calculate the absolute value of.
 * @return The absolute value of <code>x</code>.
 */ 
public <T extends Number> T abs(T x){ ... }

```

## References
* Help - Eclipse Platform: [Java Compiler Javadoc Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-javadoc.htm).
* Java SE Documentation: [How to Write Doc Comments for the Javadoc Tool](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html#@param), [Documentation Comment Specification for the Standard Doclet](https://docs.oracle.com/en/java/javase/11/docs/specs/doc-comment-spec.html#param)
