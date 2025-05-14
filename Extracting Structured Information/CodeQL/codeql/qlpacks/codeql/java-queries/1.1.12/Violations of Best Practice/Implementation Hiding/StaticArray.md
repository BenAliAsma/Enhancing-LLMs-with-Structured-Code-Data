# Array constant vulnerable to change
Constant values are typically represented by public, static, final fields. When defining several related constants, it is sometimes tempting to define a public, static, final field with an array type, and initialize it with a list of all the different constant values.

However, the `final` keyword applies only to the field itself (that is, the array reference), and not to the contents of the array. This means that the field always refers to the same array instance, but each element of the array may be modified freely. This possibly invalidates important assumptions of client code.


## Recommendation
Where possible, avoid declaring array constants. If there are only a few constant values, consider using a named constant for each one, or defining them in an `enum` type.

If you genuinely need to refer to a long list of constants with the same name and an index, consider replacing the array constant with a constant of type `List` to which you assign an unmodifiable collection. See the example for ways of achieving this.


## Example
In the following example, `public static final` applies only to `RGB` itself, not the constants that it contains.


```java
public class Display {
	// AVOID: Array constant is vulnerable to mutation.
	public static final String[] RGB = {
		"FF0000", "00FF00", "0000FF"
	};
	
	void f() {
		// Re-assigning the "constant" is legal.
		RGB[0] = "00FFFF";
	}
}
```
The following example shows examples of ways to declare constants that avoid this problem.


```java
// Solution 1: Extract to individual constants
public class Display {
    public static final String RED = "FF0000";
    public static final String GREEN = "00FF00";
    public static final String BLUE = "0000FF";
}

// Solution 2: Define constants using in an enum type
public enum Display
{
    RED ("FF0000"), GREEN ("00FF00"), BLUE ("0000FF");

    private String rgb;
    private Display(int rgb) {
        this.rgb = rgb;
    }
    public String getRGB(){
        return rgb;
    }
}

// Solution 3: Use an unmodifiable collection
public class Display {
    public static final List<String> RGB =
            Collections.unmodifiableList(
                    Arrays.asList("FF0000",
                            "00FF00",
                            "0000FF"));
}

// Solution 4: Use a utility method
public class Utils {
    public static <T> List<T> constList(T... values) {
        return Collections.unmodifiableList(
                Arrays.asList(values));
    }
}

public class Display {
    public static final List<String> RGB =
            Utils.constList("FF0000", "00FF00", "0000FF");
}

```

## References
* J. Bloch, *Effective Java (second edition)*, p. 70. Addison-Wesley, 2008.
* Java Language Specification: [4.12.4 final Variables](https://docs.oracle.com/javase/specs/jls/se11/html/jls-4.html#jls-4.12.4).
* Common Weakness Enumeration: [CWE-582](https://cwe.mitre.org/data/definitions/582.html).
