# Type variable hides another type
Type shadowing occurs if two types have the same name but one is defined within the scope of the other. This can arise if you introduce a type variable with the same name as an imported class.

Type shadowing may cause the two types to be confused, which can lead to various problems.


## Recommendation
Name the type variable so that its name does not clash with the imported class.


## Example
In the following example, the type `java.util.Map.Entry` is imported at the top of the file, but the class `Mapping` is defined with two type variables, `Key` and `Entry`. Uses of `Entry` within the `Mapping` class refer to the type variable, and not the imported interface. The type variable therefore shadows `Map.Entry`.


```java
import java.util.Map;
import java.util.Map.Entry;

class Mapping<Key, Entry>  // The type variable 'Entry' shadows the imported interface 'Entry'.
{
	// ...
}
```
To fix the code, the type variable `Entry` on line 4 should be renamed.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Java Language Specification: [6.4 Shadowing and Obscuring](https://docs.oracle.com/javase/specs/jls/se11/html/jls-6.html#jls-6.4).
