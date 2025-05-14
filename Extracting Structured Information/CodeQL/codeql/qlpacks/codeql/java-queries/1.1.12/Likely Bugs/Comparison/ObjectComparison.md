# Reference equality test on java.lang.Object
Reference comparisons (`==` or `!=`) with operands where the static type is `Object` may not work as intended. Reference comparisons check if two objects are *identical*. To check if two objects are *equivalent*, use `Object.equals` instead.


## Recommendation
Use `Object.equals` instead of `==` or `!=`, and override the default behavior of the method in a subclass, so that it uses the appropriate notion of equality.


## References
* Java API Specification: [Object.equals()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/Object.html#equals(java.lang.Object)).
* Common Weakness Enumeration: [CWE-595](https://cwe.mitre.org/data/definitions/595.html).
