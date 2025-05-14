# Implicit import
Imports can be categorized as *explicit* (for example `import java.util.List;`) or *implicit* (also known as 'on-demand', for example `import java.util.*;`):

* Implicit imports give access to all visible types in the type (or package) that precedes the ".\*"; types imported in this way never shadow other types.
* Explicit imports give access to just the named type; they can shadow other types that would normally be visible through an implicit import, or through the normal package visibility rules.
It is often considered bad practice to use implicit imports. The only advantage to doing so is making the code more concise, and there are a number of disadvantages:

* The exact dependencies of a file are not visible at a glance.
* Confusing compile-time errors can be introduced if a type name is used that could originate from several implicit imports.

## Recommendation
For readability, it is recommended to use explicit imports instead of implicit imports. Many modern IDEs provide automatic functionality to help achieve this, typically under the name "Organize imports". They can also fold away the import declarations, and automatically manage imports: adding them when a particular type is auto-completed by the editor, and removing them when they are not necessary. This functionality makes implicit imports mainly redundant.


## Example
The following example uses implicit imports. This means that it is not clear to a programmer where the `List` type on line 5 is imported from.


```java
import java.util.*;  // AVOID: Implicit import statements
import java.awt.*;

public class Customers {
	public List getCustomers() {  // Compiler error: 'List' is ambiguous.
		...
	}
}
```
To improve readability, the implicit imports should be replaced by explicit imports. For example, `import java.util.*;` should be replaced by `import java.util.List;` on line 1.


## References
* Java Language Specification: [6.4.1 Shadowing](https://docs.oracle.com/javase/specs/jls/se11/html/jls-6.html#jls-6.4.1), [7.5.2 Type-Import-on-Demand Declarations](https://docs.oracle.com/javase/specs/jls/se11/html/jls-7.html#jls-7.5.2).
