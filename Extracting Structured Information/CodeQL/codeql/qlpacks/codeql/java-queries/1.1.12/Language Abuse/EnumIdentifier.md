# String 'enum' used as identifier
Enumerations, or enums, were introduced in Java 5, with the keyword `enum`. Code written before this may use `enum` as an identifier. To compile such code, you must compile it with a command such as `javac -source 1.4 ...`. However, this means that you cannot use any new features that are provided in Java 5 and later.


## Recommendation
To make it easier to compile the code and add code that uses new Java features, rename any identifiers that are named `enum` in legacy code.


## Example
In the following example, `enum` is used as the name of a variable. This means that the code does not compile unless the compiler's source language is set to 1.4 or earlier. To avoid this constraint, the variable should be renamed.


```java
class Old
{
	public static void main(String[] args) {
		int enum = 13;  // AVOID: 'enum' is a variable.
		System.out.println("The value of enum is " + enum);
	}
}

```

## References
* Java Language Specification: [8.9 Enum Types](https://docs.oracle.com/javase/specs/jls/se11/html/jls-8.html#jls-8.9).
