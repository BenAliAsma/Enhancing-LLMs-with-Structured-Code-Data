# Non-synchronized override of synchronized method
If a synchronized method is overridden in a subclass, the compiler does not require the overriding method to be synchronized. However, if the overriding method is not synchronized, the thread-safety of the subclass may be broken.


## Recommendation
Ensure that the overriding method is synchronized, if necessary.


## References
* Java Language Specification: [Synchronization](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html#jls-17.1).
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-820](https://cwe.mitre.org/data/definitions/820.html).
