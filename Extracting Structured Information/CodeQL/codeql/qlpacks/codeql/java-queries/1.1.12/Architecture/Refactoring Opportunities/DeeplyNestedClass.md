# Deeply-nested class
Classes (especially complex ones) that are nested multiple levels deep can be difficult to understand because they have access to variables from all of the classes that enclose them. Such classes can also be difficult to unit test. Specific exceptions are made for:

* Anonymous classes - these are generally used as a substitute for closures.
* Enumerations, and simple classes that contain no methods - these are unlikely to hinder readability.

## Recommendation
The solution is to move one or more of the nested classes into a higher scope, less deeply-nested (see example below). When you move a nested class, you must:

* Ensure that the class can still access the required variables from its previously enclosing scopes.
* Consider the dependencies, particularly when you move a non-static nested class out of the containing class. Generally, a non-static class should be refactored to depend only on the contents of the classes that previously enclosed it. This avoids introducing a dependency cycle where the non-static class depends on the previously-enclosing classes themselves.

## Example
In the following example `Z1` is difficult to read because it is deeply nested.


```java
class X1 {
	private int i;

	public X1(int i) {
		this.i = i;
	}

	public Y1 makeY1(float j) {
		return new Y1(j);
	}

	class Y1 {
		private float j;

		public Y1(float j) {
			this.j = j;
		}

		public Z1 makeZ1(double k) {
			return new Z1(k);
		}

		// Violation
		class Z1 {
			private double k;

			public Z1(double k) {
				this.k = k;
			}

			public void foo() {
				System.out.println(i * j * k);
			}
		}
	}
}
public class DeeplyNestedClass {
	public static void main(String[] args) {
		new X1(23).makeY1(9.0f).makeZ1(84.0).foo();
	}
}
```
In this example, there are no nested classes and you can clearly see which variables affect which class.


```java
class X2 {
	private int i;

	public X2(int i) {
		this.i = i;
	}

	public Y2 makeY2(float j) {
		return new Y2(i, j);
	}
}

class Y2 {
	private int i;
	private float j;

	public Y2(int i, float j) {
		this.i = i;
		this.j = j;
	}

	public Z2 makeZ2(double k) {
		return new Z2(i, j, k);
	}
}

class Z2 {
	private int i;
	private float j;
	private double k;

	public Z2(int i, float j, double k) {
		this.i = i;
		this.j = j;
		this.k = k;
	}

	public void foo() {
		System.out.println(i * j * k);
	}
}

public class NotNestedClass {
	public static void main(String[] args) {
		new X2(23).makeY2(9.0f).makeZ2(84.0).foo();
	}
}
```

## References
* The Java Tutorials: [Nested Classes](https://docs.oracle.com/javase/tutorial/java/javaOO/nested.html).
