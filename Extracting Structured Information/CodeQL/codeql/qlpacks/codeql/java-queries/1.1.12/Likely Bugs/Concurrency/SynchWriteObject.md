# Inconsistent synchronization for writeObject()
Classes with a synchronized `writeObject` method but no other synchronized methods usually lack a sufficient level of synchronization. If any mutable state of this class can be modified without proper synchronization, the serialization using the `writeObject` method may result in an inconsistent state.


## Recommendation
See if synchronization is necessary on methods other than `writeOject` to make the class thread-safe. Any methods that access or modify the state of an object of this class should usually be synchronized as well.


## References
* Java Language Specification: [Synchronization](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html#jls-17.1).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
