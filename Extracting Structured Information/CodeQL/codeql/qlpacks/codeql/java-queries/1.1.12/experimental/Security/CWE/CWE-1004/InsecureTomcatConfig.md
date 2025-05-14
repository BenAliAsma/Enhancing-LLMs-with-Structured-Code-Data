# Tomcat config disables 'HttpOnly' flag (XSS risk)
When you add an application to a Tomcat server, it will generate a new `JSESSIONID` when you call `request.getSession()` or if you invoke a JSP from a servlet. If cookies are generated without the `HttpOnly` flag, an attacker can use a cross-site scripting (XSS) attack to get another user's session ID.


## Recommendation
Tomcat version 7+ automatically sets an `HttpOnly` flag on all session cookies to prevent client side scripts from accessing the session ID. In most situations, you should not override this behavior.


## Example
The following example shows a Tomcat configuration with `useHttpOnly` disabled. Usually you should not set this.


```xml
<web-app xmlns="http://java.sun.com/xml/ns/javaee" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://java.sun.com/xml/ns/javaee
          http://java.sun.com/xml/ns/javaee/web-app_2_5.xsd" version="2.5">
    <display-name>Sample Tomcat Web Application</display-name>
    <context-param>
        <param-name>useHttpOnly</param-name>
        <param-value>false</param-value>
    </context-param>
</web-app>
```

## References
* CWE: [Sensitive Cookie Without 'HttpOnly' Flag](https://cwe.mitre.org/data/definitions/1004.html).
* OWASP: [ HttpOnly ](https://www.owasp.org/index.php/HttpOnly).
* Common Weakness Enumeration: [CWE-1004](https://cwe.mitre.org/data/definitions/1004.html).
