# Subtle call to inherited method
If a call is made to a method from an inner class A, and a method of that name is defined in both a superclass of A and an outer class of A, it is not clear to a programmer which method is intended to be called.


## Example
In the following example, it is not clear whether the call to `printMessage` calls the method that is defined in `Outer` or `Super`.


```java
public class Outer
{
	void printMessage() {
		System.out.println("Outer");
	}
	
	class Inner extends Super
	{
		void ambiguous() {
			printMessage();  // Ambiguous call
		}
	}
	
	public static void main(String[] args) {
		new Outer().new Inner().ambiguous();
	}
}

class Super
{
	void printMessage() {
		System.out.println("Super");
	}
}

```
Inherited methods take precedence over methods in outer classes, so the method in the superclass is called. However, such situations are a potential cause of confusion and defects.


## Recommendation
Resolve the ambiguity by explicitly qualifying the method call:

* To specify the outer class, prefix the method with `Outer.this.`.
* To specify the superclass, prefix the method with `super.`.
In the above example, the call to `printMessage` could be replaced by either `Outer.this.printMessage` or `super.printMessage`, depending on which method you intend to call. To preserve the behavior in the example, use `super.printMessage`.


## References
* Inner Classes Specification: [What are top-level classes and inner classes?](http://www.cis.upenn.edu/~bcpierce/courses/629/jdkdocs/guide/innerclasses/spec/innerclasses.doc1.html).
