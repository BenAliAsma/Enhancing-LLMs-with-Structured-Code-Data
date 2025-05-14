# Cast of 'this' to a type parameter
Casting `this` to a type parameter usually suggests that there is an implicit type constraint - the programmer probably wanted to express the notion that `this` could be converted to the type parameter (when using the enclosing method from derived types). However, casting to the desired type, relies on derived types to ensure that the cast will succeed without the compiler forcing them to do so.


## Recommendation
The solution is to enforce the constraint by adding an abstract method on the base type (see example below). Each derived type must then implement this method, which makes the constraint checkable by the compiler and removes the need for a cast.


## Example
In this example `BadBaseNode` relies on derived types to use the right pattern.


```java
public class CastThisToTypeParameter {
	private abstract static class BadBaseNode<T extends BadBaseNode<T>> {
		public abstract T getParent();

		public T getRoot() {
			// BAD: relies on derived types to use the right pattern
			T cur = (T)this;
			while(cur.getParent() != null) {
				cur = cur.getParent();
			}
			return cur;
		}
	}
}
```
This constraint is better enforced by adding an abstract method on the base type. Implementing this method makes the constraint checkable by the compiler.


```java
public class CastThisToTypeParameter {
	private abstract static class GoodBaseNode<T extends GoodBaseNode<T>> {
		public abstract T getSelf();
		public abstract T getParent();

		public T getRoot() {
			// GOOD: introduce an abstract method to enforce the constraint
			// that 'this' can be converted to T for derived types
			T cur = getSelf();
			while(cur.getParent() != null)
			{
				cur = cur.getParent();
			}
			return cur;
		}
	}

	private static class GoodConcreteNode extends GoodBaseNode<GoodConcreteNode> {
		private String name;
		private GoodConcreteNode parent;

		public GoodConcreteNode(String name, GoodConcreteNode parent)
		{
			this.name = name;
			this.parent = parent;
		}

		@Override
		public GoodConcreteNode getSelf() {
			return this;
		}

		@Override
		public GoodConcreteNode getParent() {
			return parent;
		}

		@Override
		public String toString() {
			return name;
		}
	}

	public static void main(String[] args) {
		GoodConcreteNode a = new GoodConcreteNode("a", null);
		GoodConcreteNode b = new GoodConcreteNode("b", a);
		GoodConcreteNode c = new GoodConcreteNode("c", a);
		GoodConcreteNode d = new GoodConcreteNode("d", b);
		GoodConcreteNode root = d.getRoot();
		System.out.println(a + " " + root);
	}
}
```
