# Jakarta Expression Language injection
Jakarta Expression Language (EL) is an expression language for Java applications. There is a single language specification and multiple implementations such as Glassfish, Juel, Apache Commons EL, etc. The language allows invocation of methods available in the JVM. If an expression is built using attacker-controlled data, and then evaluated, it may allow the attacker to run arbitrary code.


## Recommendation
It is generally recommended to avoid using untrusted data in an EL expression. Before using untrusted data to build an EL expression, the data should be validated to ensure it is not evaluated as expression language. If the EL implementation offers configuring a sandbox for EL expressions, they should be run in a restrictive sandbox that allows accessing only explicitly allowed classes. If the EL implementation does not support sandboxing, consider using other expression language implementations with sandboxing capabilities such as Apache Commons JEXL or the Spring Expression Language.


## Example
The following example shows how untrusted data is used to build and run an expression using the JUEL interpreter:


```java
String expression = "${" + getRemoteUserInput() + "}";
ExpressionFactory factory = new de.odysseus.el.ExpressionFactoryImpl();
ValueExpression e = factory.createValueExpression(context, expression, Object.class);
SimpleContext context = getContext();
Object result = e.getValue(context);
```
JUEL does not support running expressions in a sandbox. To prevent running arbitrary code, incoming data has to be checked before including it in an expression. The next example uses a Regex pattern to check whether a user tries to run an allowed expression or not:


```java
String input = getRemoteUserInput();
String pattern = "(inside|outside)\\.(temperature|humidity)";
if (!input.matches(pattern)) {
    throw new IllegalArgumentException("Unexpected expression");
}
String expression = "${" + input + "}";
ExpressionFactory factory = new de.odysseus.el.ExpressionFactoryImpl();
ValueExpression e = factory.createValueExpression(context, expression, Object.class);
SimpleContext context = getContext();
Object result = e.getValue(context);

```

## References
* Eclipse Foundation: [Jakarta Expression Language](https://projects.eclipse.org/projects/ee4j.el).
* Jakarta EE documentation: [Jakarta Expression Language API](https://javadoc.io/doc/jakarta.el/jakarta.el-api/latest/index.html)
* OWASP: [Expression Language Injection](https://owasp.org/www-community/vulnerabilities/Expression_Language_Injection).
* JUEL: [Home page](http://juel.sourceforge.net)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
