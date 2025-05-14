# Possible confusion of local and field
If a method declares a local variable with the same name as a field, then it is very easy to mix up the two when reading or modifying the program.


## Recommendation
Consider using different names for the field and local variable to make the difference between them clear.


## Example
The following example shows a local variable `values` that has the same name as a field.


```java
public class Container
{
	private int[] values; // Field called 'values'
	
	public Container (int... values) {
		this.values = values;
	}

	public Container dup() {
		int length = values.length;
		int[] values = new int[length];  // Local variable called 'values'
		Container result = new Container(values);
		return result;
	}
}

```

## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [ 6.4 Shadowing and Obscuring](https://docs.oracle.com/javase/specs/jls/se11/html/jls-6.html#jls-6.4).
