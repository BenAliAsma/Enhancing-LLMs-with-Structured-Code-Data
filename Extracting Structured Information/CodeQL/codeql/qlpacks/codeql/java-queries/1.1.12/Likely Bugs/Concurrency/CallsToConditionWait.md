# Wait on condition
Calling `wait` on an object of type `java.util.concurrent.locks.Condition` may result in unexpected behavior because `wait` is a method of the `Object` class, not the `Condition` interface itself. Such a call is probably a typographical error: typing "wait" instead of "await".


## Recommendation
Instead of `Object.wait`, use one of the `Condition.await` methods.


## References
* Java API Specification: [java.util.concurrent.Condition](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/Condition.html).
* Common Weakness Enumeration: [CWE-662](https://cwe.mitre.org/data/definitions/662.html).
