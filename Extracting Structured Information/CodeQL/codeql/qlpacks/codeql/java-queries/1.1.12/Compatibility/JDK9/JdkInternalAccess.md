# Access to unsupported JDK-internal API
Java 9 removes access to various unsupported JDK-internal APIs by default.


## Recommendation
Examine the use of unsupported JDK-internal APIs and consider replacing them with supported APIs as recommended in the references.


## References
* Oracle JDK Documentation: [Oracle JDK 9 Migration Guide](https://docs.oracle.com/javase/9/migrate/toc.htm).
* OpenJDK Documentation: [Java Dependency Analysis Tool](https://wiki.openjdk.java.net/display/JDK8/Java+Dependency+Analysis+Tool), [JEP 260: Encapsulate Most Internal APIs](https://openjdk.java.net/jeps/260), [JEP 261: Module System](https://openjdk.java.net/jeps/261).
