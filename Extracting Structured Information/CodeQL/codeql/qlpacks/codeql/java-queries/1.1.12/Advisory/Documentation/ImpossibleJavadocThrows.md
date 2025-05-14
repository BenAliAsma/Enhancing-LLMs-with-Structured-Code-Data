# Javadoc has impossible 'throws' tag
A Javadoc `@throws` or `@exception` tag that references an exception that cannot be thrown is misleading.


## Recommendation
Ensure that you only include the `@throws` or `@exception` tags in Javadoc when an exception can be thrown.


## Example
The following example shows a method with Javadoc that claims it can throw `Exception`. Since `Exception` is a checked exception and the method does not declare that it may throw an exception, the Javadoc is wrong and should be updated.


```java
/**
 * Javadoc for method.
 *
 * @throws Exception if a problem occurs.
 */
public void noThrow() {
	System.out.println("This method does not throw.");
}
```
In the following example the Javadoc has been corrected by removing the `@throws` tag.


```java
/**
 * Javadoc for method.
 */
public void noThrow() {
	System.out.println("This method does not throw.");
}
```

## References
* Java SE Documentation: [How to Write Doc Comments for the Javadoc Tool](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html#throwstag),
