# Overloaded compareTo
Classes that implement `Comparable<T>` and define a `compareTo` method whose parameter type is not `T` *overload* the `compareTo` method instead of *overriding* it. This may not be intended.


## Example
In the following example, the call to `compareTo` on line 17 calls the method defined in class `Super`, instead of the method defined in class `Sub`, because the type of `a` and `b` is `Super`. This may not be the method that the programmer intended.


```java
public class CovariantCompareTo {
	static class Super implements Comparable<Super> {
		public int compareTo(Super rhs) {
			return -1;
		}
	}
	
	static class Sub extends Super {
		public int compareTo(Sub rhs) {  // Definition of compareTo uses a different parameter type
			return 0;
		}
	}
	
	public static void main(String[] args) {
		Super a = new Sub();
		Super b = new Sub();
		System.out.println(a.compareTo(b));
	}
}
```

## Recommendation
To *override* the `Comparable<T>.compareTo` method, the parameter of `compareTo` must have type `T`.

In the example above, this means that the type of the parameter of `Sub.compareTo` should be changed to `Super`.


## References
* J. Bloch, *Effective Java (second edition)*, Item 12. Addison-Wesley, 2008.
* Java Language Specification: [Overriding (by Instance Methods)](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.8.1), [Overloading](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4.9).
* The Java Tutorials: [Overriding and Hiding Methods](https://docs.oracle.com/javase/tutorial/java/IandI/override.html).
