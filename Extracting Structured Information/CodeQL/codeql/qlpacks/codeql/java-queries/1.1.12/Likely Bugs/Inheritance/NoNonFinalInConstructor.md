# Non-final method invocation in constructor
If a constructor calls a method that is overridden in a subclass, it can cause the overriding method in the subclass to be called before the subclass has been initialized. This can lead to unexpected results.


## Recommendation
Do not call a non-final method from within a constructor if that method could be overridden in a subclass.


## Example
In the following example, executing `new Sub("test")` results in a `NullPointerException`. This is because the subclass constructor implicitly calls the superclass constructor, which in turn calls the overridden `init` method before the field `s` is initialized in the subclass constructor.


```java
public class Super {
	public Super() {
		init();
	}
	
	public void init() {
	}
}

public class Sub extends Super {
	String s;
	int length;

	public Sub(String s) {
		this.s = s==null ? "" : s;
	}
	
	@Override
	public void init() {
		length = s.length();
	}
}
```
To avoid this problem:

* The `init` method in the super constructor should be made `final` or `private`.
* The initialization that is performed in the overridden `init` method in the subclass can be moved to the subclass constructor itself, or delegated to a separate final or private method that is called from within the subclass constructor.

## References
* J. Bloch, *Effective Java (second edition)*, pp. 89&ndash;90. Addison-Wesley, 2008.
* The Java Tutorials: [Writing Final Classes and Methods](https://docs.oracle.com/javase/tutorial/java/IandI/final.html).
