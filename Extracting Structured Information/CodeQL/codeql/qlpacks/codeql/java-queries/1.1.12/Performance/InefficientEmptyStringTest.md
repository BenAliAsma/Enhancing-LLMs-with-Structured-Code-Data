# Inefficient empty string test
When checking whether a string `s` is empty, perhaps the most obvious solution is to write something like `s.equals("")` (or `"".equals(s)`). However, this actually carries a fairly significant overhead, because `String.equals` performs a number of type tests and conversions before starting to compare the content of the strings.


## Recommendation
The preferred way of checking whether a string `s` is empty is to check if its length is equal to zero. Thus, the condition is `s.length() == 0`. The `length` method is implemented as a simple field access, and so should be noticeably faster than calling `equals`.

Note that in Java 6 and later, the `String` class has an `isEmpty` method that checks whether a string is empty. If the codebase does not need to support Java 5, it may be better to use that method instead.


## Example
In the following example, class `InefficientDBClient` uses `equals` to test whether the strings `user` and `pw` are empty. Note that the test `"".equals(pw)` guards against `NullPointerException`, but the test `user.equals("")` throws a `NullPointerException` if `user` is `null`.

In contrast, the class `EfficientDBClient` uses `length` instead of `equals`. The class preserves the behavior of `InefficientDBClient` by guarding `pw.length() == 0` but not `user.length() == 0` with an explicit test for `null`. Whether or not this guard is desirable depends on the intended behavior of the program.


```java
// Inefficient version
class InefficientDBClient {
	public void connect(String user, String pw) {
		if (user.equals("") || "".equals(pw))
			throw new RuntimeException();
		...
	}
}

// More efficient version
class EfficientDBClient {
	public void connect(String user, String pw) {
		if (user.length() == 0 || (pw != null && pw.length() == 0))
			throw new RuntimeException();
		...
	}
}

```

## References
* Java API Specification: [String.length()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#length()), [String.isEmpty()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#isEmpty()), [String.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#equals(java.lang.Object)).
