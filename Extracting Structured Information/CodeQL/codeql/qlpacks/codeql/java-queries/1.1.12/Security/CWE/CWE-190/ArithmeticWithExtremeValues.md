# Use of extreme values in arithmetic expression
Assigning the maximum or minimum value for a type to a variable of that type and then using the variable in calculations may cause overflows.


## Recommendation
Before using the variable, ensure that it is reassigned a value that does not cause an overflow, or use a wider type to do the arithmetic.


## Example
In this example, assigning `Long.MAX_VALUE` to a variable and adding one causes an overflow. However, casting to a `long` beforehand ensures that the arithmetic is done in the wider type, and so does not overflow.


```java
class Test {
	public static void main(String[] args) {	
		{
			long i = Long.MAX_VALUE;
			// BAD: overflow
			long j = i + 1;
		}
		
		{
			int i = Integer.MAX_VALUE;
			// GOOD: no overflow
			long j = (long)i + 1;
		}
	}
}
```

## References
* SEI CERT Oracle Coding Standard for Java: [NUM00-J. Detect or prevent integer overflow](https://wiki.sei.cmu.edu/confluence/display/java/NUM00-J.+Detect+or+prevent+integer+overflow).
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
* Common Weakness Enumeration: [CWE-191](https://cwe.mitre.org/data/definitions/191.html).
