# Magic numbers
A *magic number* is a numeric literal (for example, `8080`, `2048`) that is used in the middle of a block of code without explanation. It is considered bad practice to use magic numbers because:

* A number in isolation can be difficult for other programmers to understand.
* It can be difficult to update the code if the requirements change. For example, if the magic number represents the number of guests allowed, adding one more guest means that you must change every occurrence of the magic number.

## Recommendation
Assign the magic number to a new named constant, and use this instead. This overcomes the two problems with magic numbers:

* A named constant (such as `MAX_GUESTS`) is more easily understood by other programmers.
* Using the same named constant in many places makes the code much easier to update if the requirements change, because you have to update the number in only one place.

## Example
The following example shows a magic number `timeout`. This should be replaced by a new named constant, as shown in the fixed version.


```java
// Problem version
public class MagicConstants
{
	public static final String IP = "127.0.0.1";
	public static final int PORT = 8080;
	public static final String USERNAME = "test";

	public void serve(String ip, int port, String user, int timeout) {
		// ...
	}

	public static void main(String[] args) {
		int timeout = 60000;  // AVOID: Magic number

		new MagicConstants().serve(IP, PORT, USERNAME, timeout);
	}
}


// Fixed version
public class MagicConstants
{
	public static final String IP = "127.0.0.1";
	public static final int PORT = 8080;
	public static final String USERNAME = "test";
	public static final int TIMEOUT = 60000;  // Magic number is replaced by named constant

	public void serve(String ip, int port, String user, int timeout) {
		// ...
	}

	public static void main(String[] args) {
		new MagicConstants().serve(IP, PORT, USERNAME, TIMEOUT);  // Use 'TIMEOUT' constant
	}
}
```

## References
* R. C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, &sect;17.G25. Prentice Hall, 2008.
