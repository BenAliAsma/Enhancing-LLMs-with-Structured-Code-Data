# Useless upcast
In most situations, casting an instance of a derived type to a base type serves no purpose, since the conversion can be done implicitly. In such cases, the redundant cast can simply be removed. However, an upcast is not redundant in the following situations:

* It is being used to force a call to an overloaded callable that takes a parameter of the base type rather than one of the derived type.
* It is being used to specify the type to use for the operands of a ternary expression.
Both of these special cases are illustrated in the example below. This rule ignores these special cases and highlights upcasts which appear to be redundant.


## Recommendation
Remove the unnecessary upcast to simplify the code.


## Example
The following code includes an example of a redundant upcast that would be highlighted by this rule. In addition, three examples of upcasts that are required and are ignored by this rule.


```java
public class UselessUpcast {
	private static class B {}
	private static class D extends B {}

	private static void Foo(B b) { System.out.println("Foo(B)"); }
	private static void Foo(D d) { System.out.println("Foo(D)"); }

	private static class Expr {}
	private static class AddExpr extends Expr {}
	private static class SubExpr extends Expr {}

	public static void main(String[] args) {
		D d = new D();
		B b_ = (B)d;	// violation: redundant cast, consider removing

		B b = new D();
		D d_ = (D)b;	// non-violation: required downcast

		Foo(d);
		Foo((B)d);		// non-violation: required to call Foo(B)

		// Non-violation: required to specify the type of the ternary operands.
		Expr e = d != null ? (Expr)new AddExpr() : new SubExpr();
	}
}
```
