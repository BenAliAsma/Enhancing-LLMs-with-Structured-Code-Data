# Pointless forwarding method
If a class contains two distinct methods of the same name such that:

1. One method is only ever called from the other method.
1. The calling method calls only the other method and nothing else.
Then the first method is no more than a forwarding method for the second and the two methods can probably be merged.

There are several advantages to doing this:

* It reduces the cognitive overhead involved in keeping track of the various different overloaded forms of a method.
* If both methods are public, it simplifies the API of their containing class, making it more discoverable to other programmers.
* It makes it clearer to other programmers that certain methods are called and other methods are not.

## Example
In this example, the two `print` methods in `Bad` can be merged, as one is simply a forwarder for the other. The two classes `Better1` and `Better2` show two alternative ways of merging the methods.


```java
import static java.lang.System.out;

public class PointlessForwardingMethod {
	private static class Bad {
		// Violation: This print does nothing but forward to the other one, which is not
		// called independently.
		public void print(String firstName, String lastName) {
			print(firstName + " " + lastName);
		}

		public void print(String fullName) {
			out.println("Pointless forwarding methods are bad, "  + fullName + "...");
		}
	}

	private static class Better1 {
		// Better: Merge the methods, using local variables to replace the parameters in
		// the original version.
		public void print(String firstName, String lastName) {
			String fullName = firstName + " " + lastName;
			out.println("Pointless forwarding methods are bad, " + fullName + "...");
		}
	}

	private static class Better2 {
		// Better: If there's no complicated logic, you can often remove the extra
		// variables entirely.
		public void print(String firstName, String lastName) {
			out.println(
				"Pointless forwarding methods are bad, " +
				firstName + " " + lastName + "..."
			);
		}
	}

	public static void main(String[] args) {
		new Bad().print("Foo", "Bar");
		new Better1().print("Foo", "Bar");
		new Better2().print("Foo", "Bar");
	}
}
```
