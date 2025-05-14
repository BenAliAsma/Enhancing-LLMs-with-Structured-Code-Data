# EJB uses threads
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> The enterprise bean must not attempt to manage threads. The enterprise bean must not attempt to start, stop, suspend, or resume a thread, or to change a thread's priority or name. The enterprise bean must not attempt to manage thread groups.

These functions are reserved for the EJB container. Allowing the enterprise bean to manage threads would decrease the container's ability to properly manage the runtime environment.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](https://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-383](https://cwe.mitre.org/data/definitions/383.html).
* Common Weakness Enumeration: [CWE-573](https://cwe.mitre.org/data/definitions/573.html).
