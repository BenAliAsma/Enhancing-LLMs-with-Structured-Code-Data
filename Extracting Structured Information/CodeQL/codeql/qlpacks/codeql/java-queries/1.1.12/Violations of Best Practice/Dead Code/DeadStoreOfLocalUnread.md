# Useless assignment to local variable
A value is assigned to a local variable, but the local variable is only read before the assignment, not after it. This means that the assignment is suspect, because the state of the local variable that it creates is never used.


## Recommendation
Ensure that you check the control and data flow in the method carefully. If a value is really not needed, consider omitting the assignment. Be careful, though: if the right-hand side has a side-effect (like performing a method call), it is important to keep this to preserve the overall behavior.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
