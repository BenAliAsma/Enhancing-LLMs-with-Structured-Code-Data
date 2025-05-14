# Finally block may not complete normally
A `finally` block that does not complete normally suppresses any exceptions that may have been thrown in the corresponding `try` block. This can happen if the `finally` block contains any `return` or `throw` statements, or if it contains any `break` or `continue` statements whose jump target lies outside of the `finally` block.


## Recommendation
To avoid suppressing exceptions that are thrown in a `try` block, design the code so that the corresponding `finally` block always completes normally. Remove any of the following statements that may cause it to terminate abnormally:

* `return`
* `throw`
* `break`
* `continue`

## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 36. Addison-Wesley, 2005.
* Java Language Specification: [Execution of try-finally and try-catch-finally](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.20.2).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-584](https://cwe.mitre.org/data/definitions/584.html).
