# Directories and files exposure
Enabling directory listing in J2EE application servers introduces the vulnerability of filename and path disclosure, which could allow an attacker to read arbitrary files in the server web directory. This includes application source code and data, as well as credentials for back-end systems.

The query detects insecure configuration by validating its web configuration.


## Recommendation
Always disabling directory listing in the production environment.


## Example
The following two examples show two ways of directory listing configuration. In the 'BAD' case, it is enabled. In the 'GOOD' case, it is disabled.


```xml
<?xml version="1.0" encoding="UTF-8" ?>
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
                      http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd" version="4.0">

    <!-- The default servlet for all web applications, that serves static     -->
    <!-- resources.  It processes all requests that are not mapped to other   -->
    <!-- servlets with servlet mappings (defined either here or in your own   -->
    <!-- web.xml file).                                                       -->
    <servlet>
        <servlet-name>default</servlet-name>
        <servlet-class>org.apache.catalina.servlets.DefaultServlet</servlet-class>
        <init-param>
            <param-name>listings</param-name>
            <!-- GOOD: Don't allow directory listing -->
            <param-value>false</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>

    <servlet>
        <servlet-name>default</servlet-name>
        <servlet-class>org.apache.catalina.servlets.DefaultServlet</servlet-class>
        <init-param>
            <param-name>listings</param-name>
            <!-- BAD: Allow directory listing -->
            <param-value>true</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>
</web-app>
```

## References
* [CWE-548: Exposure of Information Through Directory Listing](https://cwe.mitre.org/data/definitions/548.html) [Directory listing](https://portswigger.net/kb/issues/00600100_directory-listing) [Directory traversal](https://portswigger.net/web-security/file-path-traversal)
* Common Weakness Enumeration: [CWE-548](https://cwe.mitre.org/data/definitions/548.html).
