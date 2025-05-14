# Misnamed static final field
A static, final field name that contains lowercase letters does not follow standard naming conventions, which decreases code readability. For example, `Min_Width`.


## Recommendation
Use uppercase letters throughout a static, final field name, and use underscores to separate words within the field name. For example, `MIN_WIDTH`.


## References
* J. Bloch, *Effective Java (second edition)*, Item 56. Addison-Wesley, 2008.
* Java Language Specification: [6.1. Declarations](https://docs.oracle.com/javase/specs/jls/se11/html/jls-6.html#jls-6.1).
* Java SE Documentation: [9 - Naming Conventions](https://www.oracle.com/java/technologies/javase/codeconventions-namingconventions.html).
