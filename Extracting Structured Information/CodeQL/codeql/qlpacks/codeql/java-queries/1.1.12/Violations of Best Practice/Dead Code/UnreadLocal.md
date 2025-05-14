# Unread local variable
A local variable that is never read is useless.

As a matter of good practice, there should be no unused or useless code. It makes the program more difficult to understand and maintain, and can waste a programmer's time.


## Recommendation
This rule applies to variables that are never used as well as variables that are only written to but never read. In both cases, ensure that no operations are missing that would use the local variable. If appropriate, simply remove the declaration. However, if the variable is written to, ensure that any side-effects in the assignments are retained. (For further details, see the example.)


## Example
In the following example, the local variable `oldQuantity` is assigned a value but never read. In the fixed version of the example, the variable is removed but the call to `items.put` in the assignment is retained.


```java
// Version containing unread local variable
public class Cart {
	private Map<Item, Integer> items = ...;
	public void add(Item i) {
		Integer quantity = items.get(i);
		if (quantity = null)
			quantity = 1;
		else
			quantity++;
		Integer oldQuantity = items.put(i, quantity);  // AVOID: Unread local variable
	}
}

// Version with unread local variable removed
public class Cart {
	private Map<Item, Integer> items = ...;
	public void add(Item i) {
		Integer quantity = items.get(i);
		if (quantity = null)
			quantity = 1;
		else
			quantity++;
		items.put(i, quantity);
	}
}
```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
