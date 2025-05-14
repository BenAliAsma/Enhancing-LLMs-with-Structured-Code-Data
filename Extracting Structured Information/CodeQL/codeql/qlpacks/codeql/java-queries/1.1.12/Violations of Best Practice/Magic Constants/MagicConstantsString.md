# Magic strings
A *magic string* is a string literal (for example, `"SELECT"`, `"127.0.0.1"`) that is used in the middle of a block of code without explanation. It is considered bad practice to use magic strings because:

* A string in isolation can be difficult for other programmers to understand.
* It can be difficult to update the code if the requirements change. For example, if the magic string represents a protocol, changing the protocol means that you must change every occurrence of the protocol.

## Recommendation
Assign the magic string to a new named constant, and use this instead. This overcomes the two problems with magic strings:

* A named constant (such as `SMTP_HELO`) is more easily understood by other programmers.
* Using the same named constant in many places makes the code much easier to update if the requirements change, because you have to update the string in only one place.

## Example
The following example shows a magic string `username`. This should be replaced by a new named constant, as shown in the fixed version.


```java
// Problem version
public class MagicConstants
{
	public static final String IP = "127.0.0.1";
	public static final int PORT = 8080;
	public static final int TIMEOUT = 60000;

	public void serve(String ip, int port, String user, int timeout) {
		// ...
	}

	public static void main(String[] args) {
		String username = "test";  // AVOID: Magic string

		new MagicConstants().serve(IP, PORT, username, TIMEOUT);
	}
}


// Fixed version
public class MagicConstants
{
	public static final String IP = "127.0.0.1";
	public static final int PORT = 8080;
	public static final int USERNAME = "test";  // Magic string is replaced by named constant
	public static final int TIMEOUT = 60000;

	public void serve(String ip, int port, String user, int timeout) {
		// ...
	}

	public static void main(String[] args) {
		new MagicConstants().serve(IP, PORT, USERNAME, TIMEOUT);  // Use 'USERNAME' constant
	}
}
```

## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, &sect;17.G25. Prentice Hall, 2008.
