# EJB uses synchronization
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> An enterprise bean must not use thread synchronization primitives to synchronize execution of multiple instances.

Synchronization would not work if the EJB container distributed enterprise bean's instances across multiple JVMs.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](http://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-574](https://cwe.mitre.org/data/definitions/574.html).
