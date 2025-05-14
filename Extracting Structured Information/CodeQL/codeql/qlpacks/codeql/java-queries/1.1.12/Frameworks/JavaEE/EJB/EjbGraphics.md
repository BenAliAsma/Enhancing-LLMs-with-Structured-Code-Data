# EJB uses graphics
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> An enterprise bean must not use the AWT functionality to attempt to output information to a display, or to input information from a keyboard.

Most servers do not allow direct interaction between an application program and a keyboard/display attached to the server system.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](http://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-575](https://cwe.mitre.org/data/definitions/575.html).
