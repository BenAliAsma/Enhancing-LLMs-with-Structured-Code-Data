# Unused field
A field that is neither public nor protected and never accessed is typically a leftover from old refactorings or a sign of incomplete or pending code changes.

This rule does not apply to a field in a serializable class because it may be accessed during serialization and deserialization.

Fields annotated with `@SuppressWarnings("unused")` are also not reported.


## Recommendation
If an unused field is a leftover from old refactorings, you should just remove it. If it indicates incomplete or pending code changes, finish making the changes and remove the field if it is not needed.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
