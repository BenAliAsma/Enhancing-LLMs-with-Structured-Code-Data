# Impossible array cast
Some downcasts on arrays will fail at runtime. An object `a` with dynamic type `A[]` cannot be cast to `B[]`, where `B` is a subtype of `A`, even if all the elements of `a` can be cast to `B`.


## Recommendation
Ensure that the array creation expression constructs an array object of the right type.


## Example
The following example shows an assignment that throws a `ClassCastException` at runtime.

```java
String[] strs = (String[])new Object[]{ "hello", "world" };
```
To avoid the exception, a `String` array should be created instead.

```java
String[] strs = new String[]{ "hello", "world" };
```

## References
* Java Language Specification: [Narrowing Reference Conversion](https://docs.oracle.com/javase/specs/jls/se11/html/jls-5.html#jls-5.1.6), [Subtyping among Array Types](https://docs.oracle.com/javase/specs/jls/se11/html/jls-4.html#jls-4.10.3).
* Common Weakness Enumeration: [CWE-704](https://cwe.mitre.org/data/definitions/704.html).
