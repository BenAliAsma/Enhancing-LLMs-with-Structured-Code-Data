# Unreachable catch clause
An unreachable `catch` clause may indicate a logical mistake in the exception handling code or may simply be unnecessary.

Although certain unreachable `catch` clauses cause a compiler error, there are also unreachable `catch` clauses that do not cause a compiler error. A `catch` clause `C` is considered reachable by the compiler if both of the following conditions are true:

* A checked exception that is thrown in the `try` block is assignable to the parameter of `C`.
* There is no previous `catch` clause whose parameter type is equal to, or a supertype of, the parameter type of `C`.
However, a `catch` clause that is considered reachable by the compiler can be unreachable if both of the following conditions are true:

* The `catch` clause's parameter type `E` does not include any unchecked exceptions.
* All exceptions that are thrown in the `try` block whose type is a (strict) subtype of `E` are already handled by previous `catch` clauses.

## Recommendation
Ensure that unreachable `catch` clauses are removed or that further corrections are made to make them reachable.

Note that if a `try-catch` statement contains multiple `catch` clauses, and an exception that is thrown in the `try` block matches more than one of the `catch` clauses, only the first matching clause is executed.


## Example
In the following example, the second `catch` clause is unreachable. The code is incomplete because a `FileOutputStream` is opened but no methods are called to write to the stream. Such methods typically throw `IOException`s, which would make the second `catch` clause reachable.


```java
FileOutputStream fos = null;
try {
	fos = new FileOutputStream(new File("may_not_exist.txt"));
} catch (FileNotFoundException e) {
	// ask the user and try again
} catch (IOException e) {
	// more serious, abort
} finally {
	if (fos!=null) { try { fos.close(); } catch (IOException e) { /*ignore*/ } }
}
```

## References
* Java Language Specification: [Execution of try-catch](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.20.1), [Unreachable Statements](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.21).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
