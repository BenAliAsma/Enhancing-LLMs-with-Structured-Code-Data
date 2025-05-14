# String concatenation in loop
When string concatenation is performed using the "+" operator, the compiler translates this operation to a suitable manipulation, possibly constructing several intermediate strings. In general, because strings are immutable, at least one new string has to be constructed to hold the result.

Building up a string one piece at a time in a loop requires a new string on every iteration, repeatedly copying longer and longer prefixes to fresh string objects. As a result, performance can be severely degraded.


## Recommendation
Whenever a string is constructed using a loop that iterates more than just a few times, it is usually better to create a `StringBuffer` or `StringBuilder` object and append to that. Because such buffers are based on mutable character arrays, which do not require a new string to be created for each concatenation, they can reduce the cost of repeatedly growing the string.

To choose between `StringBuffer` and `StringBuilder`, check if the new buffer object can possibly be accessed by several different threads while in use. If multi-thread safety is required, use a `StringBuffer`. For purely local string buffers, you can avoid the overhead of synchronization by using a `StringBuilder`.


## Example
The following example shows a simple test that measures the time taken to construct a string. It constructs the same string of 65,536 binary digits, character-by-character, first by repeatedly appending to a string, and then by using a `StringBuilder`. The second method is three orders of magnitude faster.


```java
public class ConcatenationInLoops {
	public static void main(String[] args) {
		Random r = new Random(123);
		long start = System.currentTimeMillis();
		String s = "";
		for (int i = 0; i < 65536; i++)
			s += r.nextInt(2);
		System.out.println(System.currentTimeMillis() - start);  // This prints roughly 4500.

		r = new Random(123);
		start = System.currentTimeMillis();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < 65536; i++)
			sb.append(r.nextInt(2));
		s = sb.toString();
		System.out.println(System.currentTimeMillis() - start);  // This prints 5.
	}
}
```

## References
* J. Bloch, *Effective Java (second edition)*, Item 51. Addison-Wesley, 2008.
* Java API Specification: [StringBuffer](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/StringBuffer.html), [StringBuilder](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/StringBuilder.html).
