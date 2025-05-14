# Test case has no tests
A JUnit 3.8 test case class (that is, a non-abstract class that is a subtype of `junit.framework.TestCase`) should contain test methods, and each method must have the correct signature to be used by JUnit. Otherwise, the JUnit test runner will fail with an error message.


## Recommendation
Ensure that the test case class contains some test methods, and that each method is of the form:

```

public void testXXX() {
  // ...
}

```
Note that the method name must start with `test` and the method must not take any parameters.

If the test case class is intended strictly for subclassing by other test case classes, consider making it abstract to document this intention. It will then no longer be flagged by this query.

This rule applies only to JUnit 3.8-style test case classes. In JUnit 4, it is no longer required to extend `junit.framework.TestCase` to mark test methods.


## Example
In the following example, `TestCaseNoTests38` does not contain any valid JUnit test methods. However, `MyTests` contains two valid JUnit test methods.


```java
// BAD: This test case class does not have any valid JUnit 3.8 test methods.
public class TestCaseNoTests38 extends TestCase {
	// This is not a test case because it does not start with 'test'.
	public void simpleTest() {
		//...
	}

	// This is not a test case because it takes two parameters.
	public void testNotEquals(int i, int j) {
		assertEquals(i != j, true);
	}

	// This is recognized as a test, but causes JUnit to fail
	// when run because it is not public.
	void testEquals() {
		//...
	}
}

// GOOD: This test case class correctly declares test methods.
public class MyTests extends TestCase {
	public void testEquals() {
		assertEquals(1, 1);
	}
	public void testNotEquals() {
		assertFalse(1 == 2);
	}
}
```

## References
* JUnit: [JUnit Cookbook](http://junit.sourceforge.net/junit3.8.1/doc/cookbook/cookbook.htm).
