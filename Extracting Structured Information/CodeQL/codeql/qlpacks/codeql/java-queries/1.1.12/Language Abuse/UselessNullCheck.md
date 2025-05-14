# Useless null check
Sometimes you can guarantee that a particular variable will never be null. For example when that variable has just been assigned a newly created object or is the exception caught by a `catch` clause. A null check on such a variable is misleading, and can potentially indicate a logic error.


## Recommendation
Do not check a variable for null if a null value is clearly impossible.


## Example
The following example shows a null check on a newly created object. An object returned by `new` can never be null, so this check is superfluous.


```java
Object o = new Object();
if (o == null) {
  // this cannot happen!
}

```

## References
* Java Language Specification: [Creation of New Class Instances](https://docs.oracle.com/javase/specs/jls/se11/html/jls-12.html#jls-12.5), [Execution of try-catch](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.20.1).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
