# Useless comparison test
The result of certain comparison tests can sometimes be inferred from their context and the results of other comparisons. This can be an indication of faulty logic and may result in dead code or infinite loops if, for example, a loop condition never changes its value.


## Recommendation
Inspect the code to check whether the logic is correct, and consider simplifying the logical expression.


## Example
In the following example the final test on `x` will always be `true`, and thus the condition is redundant and potentially wrong. If the "do more stuff" part is intended to always execute after the loop then the condition should be removed to make this clear.


```java
void method(int x) {
	while(x >= 0) {
		// do stuff
		x--;
	}
	if (x < 0) { // BAD: always true
		// do more stuff
	}
}
```

## References
* Java Language Specification: [The if Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.9).
* Common Weakness Enumeration: [CWE-570](https://cwe.mitre.org/data/definitions/570.html).
* Common Weakness Enumeration: [CWE-571](https://cwe.mitre.org/data/definitions/571.html).
