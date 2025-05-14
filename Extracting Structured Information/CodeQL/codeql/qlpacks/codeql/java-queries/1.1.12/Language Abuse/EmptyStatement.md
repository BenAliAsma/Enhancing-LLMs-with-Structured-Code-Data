# Empty statement
An *empty statement* is a single semicolon `;` that does not terminate another statement. Such a statement hinders readability and has no effect on its own.


## Recommendation
Avoid empty statements. If a loop is intended to have an empty body, it is better to mark that fact explicitly by using a pair of braces `{}` containing an explanatory comment for the body, rather than a single semicolon.


## Example
In the following example, there is an empty statement on line 3, where an additional semicolon is used. On line 6, the `for` statement has an empty body because the condition is immediately followed by a semicolon. In this case, it is better to include a pair of braces `{}` containing an explanatory comment for the body instead.


```java
public class Cart {
	// AVOID: Empty statement
	List<Item> items = new ArrayList<Cart>();;
	public void applyDiscount(float discount) {
		// AVOID: Empty statement as loop body
		for (int i = 0; i < items.size(); items.get(i++).applyDiscount(discount));
	}
}
```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
