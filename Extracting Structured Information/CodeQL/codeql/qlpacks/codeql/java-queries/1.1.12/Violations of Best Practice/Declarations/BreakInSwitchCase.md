# Unterminated switch case
In a `switch` statement, execution 'falls through' from one `case` to the next, unless the `case` ends with a `break` statement. A common programming error is to forget to insert a `break` at the end of a `case`.


## Recommendation
End each `case` with a `break` statement or, if execution is supposed to fall through to the next `case`, comment the last line of the `case` with the following comment: `/* falls through */`

Such comments are not required for a completely empty `case` that is supposed to share the same implementation with the subsequent `case`.


## Example
In the following example, the `PING` case is missing a `break` statement. As a result, after `reply` is assigned the value of `Message.PONG`, execution falls through to the `TIMEOUT` case. Then the value of `reply` is erroneously assigned the value of `Message.PING`. To fix this, insert `break;` at the end of the `PING` case.


```java
class Server
{
	public void respond(Event event)
	{
		Message reply = null;
		switch (event) {
		case PING:
			reply = Message.PONG;
			// Missing 'break' statement
		case TIMEOUT:
			reply = Message.PING;
		case PONG:
			// No reply needed
		}
		if (reply != null)
			send(reply);
	}

	private void send(Message message) {
		// ...
	}
}

enum Event { PING, PONG, TIMEOUT }
enum Message { PING, PONG }
```

## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 23. Addison-Wesley, 2005.
* Code Conventions for the Java Programming Language: [7.8 switch Statements](https://www.oracle.com/java/technologies/javase/codeconventions-statements.html#468).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-484](https://cwe.mitre.org/data/definitions/484.html).
