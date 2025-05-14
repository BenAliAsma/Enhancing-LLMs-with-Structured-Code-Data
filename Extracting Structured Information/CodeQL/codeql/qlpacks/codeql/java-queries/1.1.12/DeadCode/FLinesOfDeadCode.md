# Lines of dead code in files
Redundant, or "dead", code imposes a burden on those reading or maintaining the software project. It can make it harder to understand the structure of the code, as well as increasing the complexity of adding new features or fixing bugs. It can also affect compilation and build times for the project, as dead code will still be compiled and built even if it is never used. In some cases it may also affect runtime performance - for example, fields that are written to but never read from, where the value written to the field is expensive to compute. Removing dead code should not change the meaning of the program.

A class, method, or field may be dead even if it has dependencies from other parts of the program, if those dependencies are from code that is also considered to be dead. We can also consider this from the opposite side - an element is live, if and only if there is an entry point - such as a `main` method - that eventually calls the method, reads the field or constructs the class.

When identifying dead code, we make an assumption that the snapshot of the project includes all possible callers of the code. If the project is a library project, this may not be the case, and code may be flagged as dead when it is only used by other projects not included in the snapshot.

You can customize the results by defining additional "entry points" or by identifying fields that are accessed using reflection. You may also wish to "whitelist" classes, methods or fields that should be excluded from the results. Please refer to the Semmle documentation for more information.


## Recommendation
Any code that is marked as dead should be reviewed and, if it is genuinely not used, deleted. You can see which classes, methods and fields contribute to this metric using the rules for Dead Code analysis.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
