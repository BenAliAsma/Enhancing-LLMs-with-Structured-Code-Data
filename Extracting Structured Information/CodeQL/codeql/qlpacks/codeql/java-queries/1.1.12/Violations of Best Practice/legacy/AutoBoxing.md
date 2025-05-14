# Auto boxing or unboxing
For each primitive type, such as `int` or `double`, there is a corresponding *boxed* reference type, such as `Integer` or `Double`. These boxed versions differ from their primitive equivalents because they can hold an undefined `null` element in addition to numeric (or other) values, and there can be more than one instance of a boxed type representing the same value.

In Java 5 and later, automated boxing and unboxing conversions have been added to the language. Although these automated conversions reduce the verbosity of the code, they can hide potential problems. Such problems include performance issues because of unnecessary object creation, and confusion of boxed types with their primitive equivalents.


## Recommendation
Generally, you should use primitive types (boolean, byte, char, short, int, long, float, double) in preference to boxed types (Boolean, Byte, Character, Short, Integer, Long, Float, Double), whenever there is a choice. Exceptions are when a primitive value is used in collections and other parameterized types, or when a `null` value is explicitly used to represent an undefined value.

Where they cannot be avoided, perform boxing and unboxing conversions explicitly to avoid possible confusion of boxed types and their primitive equivalents. In cases where boxing conversions cause performance issues, use primitive types instead.


## Example
In the following example, declaring the variable `sum` to have boxed type `Long` causes it to be unboxed and reboxed during execution of the statement inside the loop.


```java
Long sum = 0L; 
for (long k = 0; k < Integer.MAX_VALUE; k++) {
	sum += k;  // AVOID: Inefficient unboxing and reboxing of 'sum'
}
```
To avoid this inefficiency, declare `sum` to have primitive type `long` instead.


## References
* J. Bloch, *Effective Java (second edition)*, Item 49. Addison-Wesley, 2008.
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [5.1.7 Boxing Conversion](https://docs.oracle.com/javase/specs/jls/se11/html/jls-5.html#jls-5.1.7).
* Java SE Documentation: [Autoboxing](https://docs.oracle.com/javase/8/docs/technotes/guides/language/autoboxing.html).
* The Java Tutorials: [Autoboxing and Unboxing](https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html).
