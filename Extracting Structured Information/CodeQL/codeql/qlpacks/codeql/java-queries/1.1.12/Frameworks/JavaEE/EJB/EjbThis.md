# EJB uses 'this' as argument or result
The Enterprise JavaBeans 3.0 core specification, Section 21.1.2, states:

> The enterprise bean must not attempt to pass `this` as an argument or method result. The enterprise bean must pass the result of `SessionContext.getBusinessObject`, `SessionContext.getEJBObject`, `SessionContext.getEJBLocalObject`, `EntityContext.getEJBObject`, or `EntityContext.getEJBLocalObject` instead.


## References
* [ JSR-220 Enterprise JavaBeans 3.0 Final Release](http://jcp.org/aboutJava/communityprocess/final/jsr220/index.html) (ejbcore), Section 21.1.2 Programming Restrictions
* Common Weakness Enumeration: [CWE-573](https://cwe.mitre.org/data/definitions/573.html).
