# Constant interface anti-pattern
Definitions of constants (meaning static, final fields) should be placed in an appropriate class where they belong logically. However, it is usually bad practice to implement an interface (or extend an abstract class) only to put a number of constant definitions into scope.


## Recommendation
The preferred way of putting the constant definitions into scope is to use the `import static` directive, which allows a compilation unit to put any visible static members from other classes into scope.

This issue is discussed in \[Bloch\]:

> That a class uses some constants internally is an implementation detail. Implementing a constant interface causes this implementation detail to leak into the classes exported API. It is of no consequence to the users of a class that the class implements a constant interface. In fact, it may even confuse them. Worse, it represents a commitment: if in a future release the class is modified so that it no longer needs to use the constants, it still must implement the interface to ensure binary compatibility.

To prevent this pollution of a class's binary interface, it is best to move the constant definitions to whatever concrete class uses them most frequently. Users of the definitions could use `import static` to access the relevant fields.


## Example
In the following example, the interface `MathConstants` has been defined only to hold a constant.


```java
public class NoConstantsOnly {
	static interface MathConstants
	{
	    public static final Double Pi = 3.14;
	}

	static class Circle implements MathConstants
	{
	    public double radius;
	    public double area()
	    {
	        return Math.pow(radius, 2) * Pi;
	    }
	}
}
```
Instead, the constant should be moved to the `Circle` class or another class that uses the constant frequently.


## References
* J. Bloch, *Effective Java (second edition)*, Item 19. Addison-Wesley, 2008.
