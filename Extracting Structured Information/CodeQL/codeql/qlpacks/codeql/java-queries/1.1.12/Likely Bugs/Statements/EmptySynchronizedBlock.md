# Empty synchronized block
Empty synchronized blocks suspend execution until a lock can be acquired, which is then released immediately. This is unlikely to achieve the desired effect and may indicate the presence of incomplete code or incorrect synchronization. It may also lead to concurrency problems.


## Recommendation
Check which code needs to be synchronized. Any code that requires synchronization on the given lock should be placed within the synchronized block.


## References
* Java Language Specification: [The synchronized Statement](https://docs.oracle.com/javase/specs/jls/se11/html/jls-14.html#jls-14.19).
* The Java Tutorials: [Synchronization](https://docs.oracle.com/javase/tutorial/essential/concurrency/sync.html).
* Common Weakness Enumeration: [CWE-585](https://cwe.mitre.org/data/definitions/585.html).
