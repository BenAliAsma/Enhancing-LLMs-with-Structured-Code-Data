# Self-containedness of files
This metric measures the percentage of the types on which a compilation unit depends for which we have source code available.

The availability of source code is one of the key factors affecting how easy or difficult it will be to build a software project in the future, especially on platforms other than those for which it was originally developed. Projects will a high level of self-containedness are likely to be more portable and easier to build in ten years' time than those that depend on many binary-only, third-party libraries. (This is one reason why many of the dependencies of open-source projects are distributed as source code, aside from the fact that the binaries are generally larger and more unwieldy to distribute.)

In the context of Java's platform independence, the availability of source code is less critical than it is for platform-dependent languages. However, note that there can be minor binary incompatibilities between different versions of Java.


## Recommendation
Low self-containedness may or may not be a problem, depending on the context of your project. However, if you determine that it is an issue for you, it is best tackled at a project level, in the following ways:

* Try to use libraries for which the source code is available.
* Try to obtain the source code for binary-only libraries from the authors.
* Where practical, rewrite parts of your code to reduce your dependence on external libraries.

## References
* Wikipedia: [Software portability](http://en.wikipedia.org/wiki/Software_portability)
* Oracle Technology Network: [Java SE 6 Compatibility](https://www.oracle.com/java/technologies/javase/compatibility.html)
* Java Language Specification: [Binary Compatibility](https://docs.oracle.com/javase/specs/jls/se11/html/jls-13.html)
