# Unnecessary cast
A cast is unnecessary if the type of the operand is already the same as the type that is being cast to.


## Recommendation
Avoid including unnecessary casts.


## Example
In the following example, casting `i` to an `Integer` is not necessary. It is already an `Integer`.


```java
public class UnnecessaryCast {
    public static void main(String[] args) {
        Integer i = 23;
        Integer j = (Integer)i;  // AVOID: Redundant cast
    }
}
```
To fix the code, delete `(Integer)` on the right-hand side of the assignment on line 4.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
