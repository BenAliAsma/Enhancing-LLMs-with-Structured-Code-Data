# TestCase implements tearDown but doesn't call super.tearDown()
A JUnit 3.8 test method that overrides a non-empty `tearDown` method should call `super.tearDown` to make sure that the superclass performs its de-initialization routines. Not calling `tearDown` may result in test failures in subsequent tests, or allow the current state to persist and affect any following tests.


## Recommendation
Call `super.tearDown` at the end of the overriding `tearDown` method.


## Example
In the following example, `TearDownNoSuper.tearDown` does not call `super.tearDown`, which may cause subsequent tests to fail, or allow the internal state to be maintained. However, `TearDownSuper.tearDown` *does* call `super.tearDown`, at the end of the method, to enable `FrameworkTestCase.tearDown` to perform de-initialization.


```java
// Abstract class that initializes then shuts down the
// framework after each set of tests
abstract class FrameworkTestCase extends TestCase {
	@Override
	protected void setUp() throws Exception {
		super.setUp();
		Framework.init();
	}
	
	@Override
	protected void tearDown() throws Exception {
		super.tearDown();
		Framework.shutdown();
	}
}

// The following classes extend 'FrameworkTestCase' to reuse the
// 'setUp' and 'tearDown' methods of the framework.

public class TearDownNoSuper extends FrameworkTestCase {
	@Override
	protected void setUp() throws Exception {
		super.setUp();
	}
	
	public void testFramework() {
		//...
	}
	
	public void testFramework2() {
		//...
	}
	
	@Override
	protected void tearDown() throws Exception {
		// BAD: Does not call 'super.tearDown'. May cause later tests to fail
		// when they try to re-initialize an already initialized framework.
		// Even if the framework allows re-initialization, it may maintain the
		// internal state, which could affect the results of succeeding tests.
		System.out.println("Tests complete");
	}
}

public class TearDownSuper extends FrameworkTestCase {
	@Override
	protected void setUp() throws Exception {
		super.setUp();
	}
	
	public void testFramework() {
		//...
	}
	
	public void testFramework2() {
		//...
	}
	
	@Override
	protected void tearDown() throws Exception {
		// GOOD: Correctly calls 'super.tearDown' to shut down the
		// framework.
		System.out.println("Tests complete");
		super.tearDown();
	}
}
```

## References
* JUnit: [JUnit Cookbook](http://junit.sourceforge.net/junit3.8.1/doc/cookbook/cookbook.htm).
