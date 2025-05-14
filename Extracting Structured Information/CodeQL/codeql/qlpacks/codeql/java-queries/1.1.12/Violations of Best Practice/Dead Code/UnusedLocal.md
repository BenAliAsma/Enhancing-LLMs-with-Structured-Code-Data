# Unused local variable
A local variable that is not accessed or initialized is typically a sign of incomplete or pending code changes.


## Recommendation
If an unused variable is no longer needed following refactoring, you should just remove it. If there are incomplete or pending code changes, finish making the changes, and then remove the variable if it is no longer needed.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
