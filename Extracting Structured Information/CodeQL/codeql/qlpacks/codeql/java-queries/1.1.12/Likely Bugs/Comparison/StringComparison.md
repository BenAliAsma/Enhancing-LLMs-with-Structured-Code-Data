# Reference equality test on strings
Comparing two `String` objects using `==` or `!=` compares object identity, which may not be intended. The same sequence of characters can be represented by two distinct `String` objects.


## Recommendation
To see if two `String` objects represent the same sequence of characters, you should usually compare the objects by using their `equals` methods.


## Example
With the following definition, `headerStyle` is compared to the empty string using `==`. This comparison can yield `false` even if `headerStyle` is the empty string, because it compares the identity of the two string objects rather than their contents. For example, if `headerStyle` was initialized by an XML parser or a JSON parser, then it might have been created with code like `String.valueOf(buf,start,len)`. Such code will produce a new string object every time it is called.


```java
void printHeader(String headerStyle) {
	if (headerStyle == null || headerStyle == "") {
		// No header
		return;
	}
	// ... print the header
}

```
With the following definition, `headerStyle` is tested using the `equals` method. This version will reliably detect whenever `headerStyle` is the empty string.


```java
void printHeader(String headerStyle) {
	if (headerStyle == null || headerStyle.equals("")) {
		// No header
		return;
	}
	// ... print the header
}

```

## References
* Java API Specification: [String.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#equals(java.lang.Object)), [String.intern()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#intern()).
* Java Language Specification: [15.21.3 Reference Equality Operators == and !=](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.21.3), [3.10.5 String Literals ](https://docs.oracle.com/javase/specs/jls/se11/html/jls-3.html#jls-3.10.5), [15.28 Constant Expressions](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.28).
* Common Weakness Enumeration: [CWE-597](https://cwe.mitre.org/data/definitions/597.html).
