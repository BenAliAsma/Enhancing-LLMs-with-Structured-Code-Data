# Dubious type test of 'this'
Testing whether `this` is an instance of a derived type creates a dependency cycle. Derived types already depend on their parent type and the cast creates a dependency in the other direction.

Dependency cycles should be avoided as they make code both difficult to read and difficult to test. In addition, a type should not know about its specific descendants, even though it may impose some constraints on them as a group (for example, the need for every derived type to implement a method with a specific signature).


## Recommendation
The base and derived types should be redesigned so that there is no need for the base type to depend on the types deriving from it.


## Example
In this example, `BadBase` introduces a dependency cycle with `Derived` by testing the type of `this`.


```java
public class DubiousTypeTestOfThis {
	private static class BadBase {
		private Derived d;

		public BadBase(Derived d) {
			if(d != null && this instanceof Derived)	// violation
				this.d = (Derived)this;
			else
				this.d = d;
		}
	}

	private static class Derived extends BadBase {
		public Derived() {
			super(null);
		}
	}

	public static void main(String[] args) {}
}
```
