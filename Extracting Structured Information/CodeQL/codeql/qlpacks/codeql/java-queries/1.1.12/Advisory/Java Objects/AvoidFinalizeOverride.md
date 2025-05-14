# Overriding definition of finalize()
Overriding the `Object.finalize` method is not a reliable way to terminate use of resources. In particular, there are no guarantees regarding the timeliness of finalizer execution.


## Recommendation
Provide explicit termination methods, which should be called by users of an API.


## References
* J. Bloch, *Effective Java (second edition)*, Item 7. Addison-Wesley, 2008.
* Java Language Specification: [12.6. Finalization of Class Instances](https://docs.oracle.com/javase/specs/jls/se11/html/jls-12.html#jls-12.6).
