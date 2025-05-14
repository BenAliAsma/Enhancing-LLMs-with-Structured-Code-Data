# EJB uses native code
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> The enterprise bean must not attempt to load a native library.

This function is reserved for the EJB container. Allowing the enterprise bean to load native code would create a security hole.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](http://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-573](https://cwe.mitre.org/data/definitions/573.html).
