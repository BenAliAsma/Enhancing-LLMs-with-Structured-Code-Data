# Dereferenced variable may be null
If a variable is dereferenced, and the variable may have a `null` value on some execution paths leading to the dereferencing, the dereferencing may result in a `NullPointerException`.

A variable may also be implicitly dereferenced if its type is a boxed primitive type, and the variable occurs in a context in which implicit unboxing occurs. Note that the conditional operator unboxes its second and third operands when one of them is a primitive type and the other is the corresponding boxed type.


## Recommendation
Ensure that the variable does not have a `null` value when it is dereferenced.


## Example
In the following example, the use of the conditional operator causes implicit unboxing, since the integer literal has type `int`. If the parameter `p` is ever `null` then a `NullPointerException` will occur.


```java
public Integer f(Integer p) {
	return true ? p : 5;
}

```
If the implicit unboxing is unintentional, it can be prevented by making sure that both branches of the conditional operator have the same type.


## References
* The Java Tutorials: [Autoboxing and Unboxing](https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html).
* Java Language Specification: [Conditional Operator ? :](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.25).
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
