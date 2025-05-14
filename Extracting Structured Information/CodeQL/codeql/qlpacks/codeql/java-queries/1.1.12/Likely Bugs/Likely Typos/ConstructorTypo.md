# Typo in constructor
A method that has the same name as its declaring type may be intended to be a constructor, not a method.


## Example
The following example shows how the singleton design pattern is often misimplemented. The programmer intends the constructor of `MasterSingleton` to be protected so that it cannot be instantiated (because the singleton instance should be retrieved using `getInstance`). However, the programmer accidentally wrote `void` in front of the constructor name, which makes it a method rather than a constructor.


```java
class MasterSingleton
{
	// ...

	private static MasterSingleton singleton = new MasterSingleton();
	public static MasterSingleton getInstance() { return singleton; }

	// Make the constructor 'protected' to prevent this class from being instantiated.
	protected void MasterSingleton() { }
}

```

## Recommendation
Ensure that methods that have the same name as their declaring type are intended to be methods. Even if they are intended to be methods, it may be better to rename them to avoid confusion.


## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 63. Addison-Wesley, 2005.
* E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design Patterns: Elements of Reusable Objection-Oriented Software*, &sect;3. Addison-Wesley Longman Publishing Co. Inc., 1995.
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [8.4 Method Declarations](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.4), [8.8 Constructor Declarations](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.8).
