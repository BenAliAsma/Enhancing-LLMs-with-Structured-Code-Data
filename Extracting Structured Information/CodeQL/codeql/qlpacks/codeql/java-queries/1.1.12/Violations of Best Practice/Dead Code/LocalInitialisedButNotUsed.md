# Local variable is initialized but not used
A local variable is initialized, but the variable is never read or written to subsequently. This suggests that the local variable is either unnecessary and should be removed, or that the value was intended to be used somewhere.


## Recommendation
Ensure that you check the control and data flow in the method carefully. If a value is really not needed, consider removing the variable. Be careful, though: if the right-hand side has a side-effect (like performing a method call), it is important to keep this to preserve the overall behavior.


## References
* Wikipedia: [Redundant code](https://en.wikipedia.org/wiki/Redundant_code).
* CERT Java Coding Standard: [MSC56-J. Detect and remove superfluous code and values](https://www.securecoding.cert.org/confluence/display/java/MSC56-J.+Detect+and+remove+superfluous+code+and+values).
* Common Weakness Enumeration: [CWE-563](https://cwe.mitre.org/data/definitions/563.html).
