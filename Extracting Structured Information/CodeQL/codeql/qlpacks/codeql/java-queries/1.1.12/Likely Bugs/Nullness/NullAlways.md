# Dereferenced variable is always null
If a variable is dereferenced, and the variable has a `null` value on all possible execution paths leading to the dereferencing, the dereferencing is guaranteed to result in a `NullPointerException`.

A variable may also be implicitly dereferenced if its type is a boxed primitive type, and the variable occurs in a context in which implicit unboxing occurs. Note that the conditional operator unboxes its second and third operands when one of them is a primitive type and the other is the corresponding boxed type.


## Recommendation
Ensure that the variable does not have a `null` value when it is dereferenced.


## Example
In the following examples, the condition `!dir.exists()` is only executed if `dir` is `null`. The second example guards the expression correctly by using `&&` instead of `||`.


```java
public void createDir(File dir) {
	if (dir != null || !dir.exists()) // BAD
		dir.mkdir();
}

public void createDir(File dir) {
	if (dir != null && !dir.exists()) // GOOD
		dir.mkdir();
}

```

## References
* The Java Tutorials: [Autoboxing and Unboxing](https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html).
* Java Language Specification: [Conditional Operator ? :](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.25).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
