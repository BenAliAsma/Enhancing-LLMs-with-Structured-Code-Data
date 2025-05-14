# Bad suite method
JUnit 3.8 requires that a `suite` method for defining a `TestSuite` that will be used by a `TestRunner` has a specific signature. If the `suite` method does not have the expected signature, JUnit does not detect the method as a `suite` method.


## Recommendation
Make sure that `suite` methods in junit `TestCase` classes are declared both `public` and `static`, and that they have a return type of `junit.framework.Test` or one of its subtypes.


## Example
In the following example, `BadSuiteMethod.suite` is not detected by JUnit because it is not declared `public`. However, `CorrectSuiteMethod.suite` *is* detected by JUnit because it has the expected signature.


```java
public class BadSuiteMethod extends TestCase {
	// BAD: JUnit 3.8 does not detect the following method as a 'suite' method.
	// The method should be public, static, and return 'junit.framework.Test' 
	// or one of its subtypes.
	static Test suite() {
		TestSuite suite = new TestSuite();
		suite.addTest(new MyTests("testEquals"));
		suite.addTest(new MyTests("testNotEquals"));
		return suite;
	}
}

public class CorrectSuiteMethod extends TestCase {
	// GOOD: JUnit 3.8 correctly detects the following method as a 'suite' method.
	public static Test suite() {
		TestSuite suite = new TestSuite();
		suite.addTest(new MyTests("testEquals"));
		suite.addTest(new MyTests("testNotEquals"));
		return suite;
	}
}
```

## References
* JUnit: [JUnit Cookbook](http://junit.sourceforge.net/junit3.8.1/doc/cookbook/cookbook.htm).
