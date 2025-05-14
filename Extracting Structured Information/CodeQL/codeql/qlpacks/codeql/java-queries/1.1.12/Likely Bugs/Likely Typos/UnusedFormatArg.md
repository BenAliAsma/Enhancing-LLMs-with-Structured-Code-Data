# Unused format argument
When formatting strings using `printf`-style format strings, one must ensure that the number of supplied arguments matches the number of arguments referenced by the format string. Additional arguments will be thrown away silently, which may not be the intended behavior, and too few arguments will cause an `IllegalFormatException`.

Format strings are used by the `format` method on the classes `String`, `Formatter`, `Console`, `PrintWriter`, and `PrintStream`. Several of these classes also supply the method alias `printf`. The class `Console` has two additional methods, `readLine` and `readPassword`, that also use format strings.


## Recommendation
Change the format string to use all the arguments, or remove the unnecessary arguments.


## Example
The following example supplies three arguments to be formatted, but the format string only refers to two arguments, so this will silently ignore the third argument.


```java
System.out.format("First string: %s Second string: %s", "Hello", "world", "!");

```

## References
* Java API Specification: [Format string syntax](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Formatter.html#syntax), [Class String](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html), [Class Formatter](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Formatter.html), [Class Console](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/Console.html), [Class PrintWriter](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/PrintWriter.html), [Class PrintStream](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/PrintStream.html).
* SLF4J library: [org.slf4j.Logger](https://www.slf4j.org/apidocs/org/slf4j/Logger.html).
* Common Weakness Enumeration: [CWE-685](https://cwe.mitre.org/data/definitions/685.html).
