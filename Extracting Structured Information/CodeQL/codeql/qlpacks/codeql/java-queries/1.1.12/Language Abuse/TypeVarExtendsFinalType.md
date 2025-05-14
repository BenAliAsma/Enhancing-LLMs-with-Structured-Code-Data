# Type bound extends a final class
A type wildcard with an `extends` clause (for example `?&nbsp;extends&nbsp;String`) implicitly suggests that a type (in this case `String`) has subclasses. If the type in the `extends` clause is final, the code is confusing because a final class cannot have any subclasses. The only type that satisfies `?&nbsp;extends&nbsp;String` is `String`.


## Recommendation
To make the code more readable, omit the wildcard to leave just the final type.


## Example
In the following example, a wildcard is used to refer to any type that is a subclass of `String`.


```java
class Printer
{
	void print(List<? extends String> strings) {  // Unnecessary wildcard
		for (String s : strings)
			System.out.println(s);
	}
}
```
However, because `String` is declared `final`, it does not have any subclasses. Therefore, it is clearer to replace `?&nbsp;extends&nbsp;String` with `String`.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [4.5.1 Type Arguments of Parameterized Types ](https://docs.oracle.com/javase/specs/jls/se11/html/jls-4.html#jls-4.5.1), [8.1.1.2 final Classes](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.1.1.2).
