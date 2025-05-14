# Empty branch of conditional, or empty loop body
An unexplained empty block or statement makes the code less readable. It might also indicate missing code, a misplaced semicolon, or a misplaced brace. For these reasons, it should be avoided.


## Recommendation
If a block is empty because some code is missing, add the code.

If an `if` statement has an empty `then` branch and a non-empty `else` branch, it may be possible to negate the condition and move the statements of the `else` branch into the `then` branch.

If a block is deliberately empty, add a comment to explain why.


## Example
In the following example, the `while` loop has intentionally been left empty. The purpose of the loop is to scan a `String` for the first occurrence of the character `'='`. A programmer reading the code might not understand the reason for the empty loop body, and think that something is missing, or perhaps even that the loop is useless. Therefore it is a good practice to add a comment to an empty block explaining why it is empty.


```java
public class Parser
{
	public void parse(String input) {
		int pos = 0;
		// ...
		// AVOID: Empty block
		while (input.charAt(pos++) != '=') { }
		// ...
	}
}

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [14.2 Blocks](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.2), [14.6 The Empty Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.6), [14.9 The if Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.9), [14.12 The while Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.12), [14.13 The do Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.13), [14.14 The for Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.14).
