# Contradictory type checks
If an `instanceof` expression occurs in a position where the type test is guaranteed to return `false`, this is often due to a typo or logical mistake. It also suggests that the surrounding code is not well tested, or possibly even dead.

Similarly, a cast that is guaranteed to fail usually indicates badly tested or dead code.


## Recommendation
Inspect the surrounding code for logical errors.


## Example
In the following example, method `getKind` first checks whether its argument `x` is an instance of class `Mammal`, and then whether it is an instance of class `Tiger`.


```java
String getKind(Animal a) {
	if (a instanceof Mammal) {
		return "Mammal";
	} else if (a instanceof Tiger) {
		return "Tiger!";
	} else {
		return "unknown";
	}
}
```
If `Tiger` is a subclass of `Mammal`, then the second `instanceof` check can never evaluate to `true`. Clearly, the two conditions should be swapped:


```java
String getKind(Animal a) {
	if (a instanceof Tiger) {
		return "Tiger!";
	} else if (a instanceof Mammal) {
		return "Mammal";
	} else {
		return "unknown";
	}
}
```

## References
* Java Language Specification: [Type Comparison Operator instanceof](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.20.2).
