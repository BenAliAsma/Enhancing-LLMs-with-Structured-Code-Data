# Missing catch of NumberFormatException
Methods such as `Integer.parseInt` that parse strings into numbers throw `NumberFormatException` if their arguments cannot be parsed. This exception should be caught so that any parse errors can be handled.


## Recommendation
It is usually best to handle `NumberFormatException` in a `catch` clause surrounding the call to the parsing method.


## Example
In the following example, the first call to `Integer.parseInt` does not catch the exception. The second call does.


```java
String s = ...;
int n;

n = Integer.parseInt(s); // BAD: NumberFormatException is not caught.

try {
        n = Integer.parseInt(s);
} catch (NumberFormatException e) {  // GOOD: The exception is caught. 
        // Handle the exception
}

```

## References
* Java API Specification: [Integer.valueOf](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Integer.html#valueOf(java.lang.String)), [Integer.parseInt](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Integer.html#parseInt(java.lang.String)), [Long.parseLong](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Long.html#parseLong(java.lang.String)), [NumberFormatException](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/NumberFormatException.html).
* Common Weakness Enumeration: [CWE-248](https://cwe.mitre.org/data/definitions/248.html).
