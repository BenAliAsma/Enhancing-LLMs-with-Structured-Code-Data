# Apache Struts development mode enabled
Turning Apache Struts' development mode configuration on while deploying applications to production environments can lead to remote code execution.


## Recommendation
An application should disable the development mode at the time of deployment.


## Example
The following example shows a \`struts.xml\` file with \`struts.devmode\` enabled.


```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE struts PUBLIC
	"-//Apache Software Foundation//DTD Struts Configuration 2.3//EN"
	"http://struts.apache.org/dtds/struts-2.3.dtd">

<struts>
    <constant name="struts.enable.DynamicMethodInvocation" value="true" />
    <constant name="struts.devMode" value="true" />
    <constant name="struts.i18n.encoding" value="utf-8" />
    <include file="login.xml" />
</struts>

```
This can be easily corrected by setting the value of the \`struts.devmode\` parameter to false.


```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE struts PUBLIC
	"-//Apache Software Foundation//DTD Struts Configuration 2.3//EN"
	"http://struts.apache.org/dtds/struts-2.3.dtd">

<struts>
    <constant name="struts.enable.DynamicMethodInvocation" value="true" />
    <constant name="struts.devMode" value="false" />
    <constant name="struts.i18n.encoding" value="utf-8"></constant>
    <include file="login.xml" />
</struts>
```

## References
* Apache Struts: [Struts development mode configuration](https://struts.apache.org/core-developers/development-mode.html)
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
