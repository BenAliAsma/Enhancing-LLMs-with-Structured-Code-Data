# EJB interferes with container operation
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> The enterprise bean must not attempt to create a class loader; obtain the current class loader; set the context class loader; set security manager; create a new security manager; stop the JVM; or change the input, output, and error streams.

These functions are reserved for the EJB container. Allowing the enterprise bean to use these functions could compromise security and decrease the container's ability to properly manage the runtime environment.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](http://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-578](https://cwe.mitre.org/data/definitions/578.html).
* Common Weakness Enumeration: [CWE-382](https://cwe.mitre.org/data/definitions/382.html).
