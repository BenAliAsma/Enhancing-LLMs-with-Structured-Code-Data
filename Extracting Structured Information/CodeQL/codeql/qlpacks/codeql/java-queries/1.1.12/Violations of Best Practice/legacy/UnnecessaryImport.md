# Unnecessary import
An `import` statement that is not necessary (because no part of the file that it is in uses any imported type) should be avoided. Although importing too many types does not affect performance, redundant `import` statements introduce unnecessary and undesirable dependencies in the code. If an imported type is renamed or deleted, the source code cannot be compiled because the `import` statement cannot be resolved.

Unnecessary `import` statements are often an indication of incomplete refactoring.


## Recommendation
Avoid including an `import` statement that is not needed. Many modern IDEs have automated support for doing this, typically under the name 'Organize imports'. This sorts the `import` statements and removes any that are not used, and it is good practice to run such a command before every commit.


## References
* Help - Eclipse Platform: [Java Compiler Errors/Warnings Preferences](https://help.eclipse.org/2020-12/advanced/content.jsp?topic=/org.eclipse.jdt.doc.user/reference/preferences/java/compiler/ref-preferences-errors-warnings.htm).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
