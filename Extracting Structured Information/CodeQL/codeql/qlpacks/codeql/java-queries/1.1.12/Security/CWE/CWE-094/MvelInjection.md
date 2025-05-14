# Expression language injection (MVEL)
MVEL is an expression language based on Java-syntax, which offers many features including invocation of methods available in the JVM. If a MVEL expression is built using attacker-controlled data, and then evaluated, then it may allow attackers to run arbitrary code.


## Recommendation
Including user input in a MVEL expression should be avoided.


## Example
In the following sample, the first example uses untrusted data to build a MVEL expression and then runs it in the default context. In the second example, the untrusted data is validated with a custom method that checks that the expression does not contain unexpected code before evaluating it.


```java
public void evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
    new InputStreamReader(socket.getInputStream()))) {
  
    String expression = reader.readLine();
    // BAD: the user-provided expression is directly evaluated
    MVEL.eval(expression);
  }
}

public void safeEvaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
    new InputStreamReader(socket.getInputStream()))) {
  
    String expression = reader.readLine();
    // GOOD: the user-provided expression is validated before evaluation
    validateExpression(expression);
    MVEL.eval(expression);
  }
}

private void validateExpression(String expression) {
  // Validate that the expression does not contain unexpected code.
  // For instance, this can be done with allow-lists or deny-lists of code patterns.
}
```

## References
* MVEL Documentation: [Language Guide for 2.0](http://mvel.documentnode.com/).
* OWASP: [Expression Language Injection](https://owasp.org/www-community/vulnerabilities/Expression_Language_Injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
