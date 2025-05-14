# Deprecated method or constructor invocation
A method (or constructor) can be marked as deprecated using either the `@Deprecated` annotation or the `@deprecated` Javadoc tag. Using a method that has been marked as deprecated is bad practice, typically for one or more of the following reasons:

* The method is dangerous.
* There is a better alternative method.
* Methods that are marked as deprecated are often removed from future versions of an API. So using a deprecated method may cause extra maintenance effort when the API is upgraded.

## Recommendation
Avoid using a method that has been marked as deprecated. Follow any guidance that is provided with the `@deprecated` Javadoc tag, which should explain how to replace the call to the deprecated method.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java API Specification: [Annotation Type Deprecated](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Deprecated.html).
* Java SE Documentation: [How and When To Deprecate APIs](https://docs.oracle.com/javase/8/docs/technotes/guides/javadoc/deprecation/deprecation.html).
* Common Weakness Enumeration: [CWE-477](https://cwe.mitre.org/data/definitions/477.html).
