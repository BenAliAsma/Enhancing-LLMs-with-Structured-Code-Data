# Expression language injection (Spring)
The Spring Expression Language (SpEL) is a powerful expression language provided by the Spring Framework. The language offers many features including invocation of methods available in the JVM. If a SpEL expression is built using attacker-controlled data, and then evaluated in a powerful context, then it may allow the attacker to run arbitrary code.

The `SpelExpressionParser` class parses a SpEL expression string and returns an `Expression` instance that can be then evaluated by calling one of its methods. By default, an expression is evaluated in a powerful `StandardEvaluationContext` that allows the expression to access other methods available in the JVM.


## Recommendation
In general, including user input in a SpEL expression should be avoided. If user input must be included in the expression, it should be then evaluated in a limited context that doesn't allow arbitrary method invocation.


## Example
The following example uses untrusted data to build a SpEL expression and then runs it in the default powerful context.


```java
public Object evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
      new InputStreamReader(socket.getInputStream()))) {

    String string = reader.readLine();
    ExpressionParser parser = new SpelExpressionParser();
    Expression expression = parser.parseExpression(string);
    return expression.getValue();
  }
}
```
The next example shows how an untrusted SpEL expression can be run in `SimpleEvaluationContext` that doesn't allow accessing arbitrary methods. However, it's recommended to avoid using untrusted input in SpEL expressions.


```java
public Object evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
      new InputStreamReader(socket.getInputStream()))) {

    String string = reader.readLine();
    ExpressionParser parser = new SpelExpressionParser();
    Expression expression = parser.parseExpression(string);
    SimpleEvaluationContext context 
        = SimpleEvaluationContext.forReadWriteDataBinding().build();
    return expression.getValue(context);
  }
}
```

## References
* Spring Framework Reference Documentation: [Spring Expression Language (SpEL)](https://docs.spring.io/spring/docs/4.2.x/spring-framework-reference/html/expressions.html).
* OWASP: [Expression Language Injection](https://owasp.org/www-community/vulnerabilities/Expression_Language_Injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
