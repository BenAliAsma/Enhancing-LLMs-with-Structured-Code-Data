# Expression language injection (JEXL)
Java EXpression Language (JEXL) is a simple expression language provided by the Apache Commons JEXL library. The syntax is close to a mix of ECMAScript and shell-script. The language allows invocation of methods available in the JVM. If a JEXL expression is built using attacker-controlled data, and then evaluated, then it may allow the attacker to run arbitrary code.


## Recommendation
It is generally recommended to avoid using untrusted input in a JEXL expression. If it is not possible, JEXL expressions should be run in a sandbox that allows accessing only explicitly allowed classes.


## Example
The following example uses untrusted data to build and run a JEXL expression.


```java
public void evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(socket.getInputStream()))) {
    
    String input = reader.readLine();
    JexlEngine jexl = new JexlBuilder().create();
    JexlExpression expression = jexl.createExpression(input);
    JexlContext context = new MapContext();
    expression.evaluate(context);
  }
}
```
The next example shows how an untrusted JEXL expression can be run in a sandbox that allows accessing only methods in the `java.lang.Math` class. The sandbox is implemented using `JexlSandbox` class that is provided by Apache Commons JEXL 3.


```java
public void evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(socket.getInputStream()))) {
    
    JexlSandbox onlyMath = new JexlSandbox(false);
    onlyMath.white("java.lang.Math");
    JexlEngine jexl = new JexlBuilder().sandbox(onlyMath).create();
      
    String input = reader.readLine();
    JexlExpression expression = jexl.createExpression(input);
    JexlContext context = new MapContext();
    expression.evaluate(context);
  }
}
```
The next example shows another way how a sandbox can be implemented. It uses a custom implementation of `JexlUberspect` that checks if callees are instances of allowed classes.


```java
public void evaluate(Socket socket) throws IOException {
  try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(socket.getInputStream()))) {
    
    JexlUberspect sandbox = new JexlUberspectSandbox();
    JexlEngine jexl = new JexlBuilder().uberspect(sandbox).create();
      
    String input = reader.readLine();
    JexlExpression expression = jexl.createExpression(input);
    JexlContext context = new MapContext();
    expression.evaluate(context);
  }

  private static class JexlUberspectSandbox implements JexlUberspect {

    private static final List<String> ALLOWED_CLASSES =
              Arrays.asList("java.lang.Math", "java.util.Random");

    private final JexlUberspect uberspect = new JexlBuilder().create().getUberspect();

    private void checkAccess(Object obj) {
      if (!ALLOWED_CLASSES.contains(obj.getClass().getCanonicalName())) {
        throw new AccessControlException("Not allowed");
      }
    }

    @Override
    public JexlMethod getMethod(Object obj, String method, Object... args) {
      checkAccess(obj);
      return uberspect.getMethod(obj, method, args);
    }

    @Override
    public List<PropertyResolver> getResolvers(JexlOperator op, Object obj) {
      checkAccess(obj);
      return uberspect.getResolvers(op, obj);
    }

    @Override
    public void setClassLoader(ClassLoader loader) {
      uberspect.setClassLoader(loader);
    }

    @Override
    public int getVersion() {
      return uberspect.getVersion();
    }

    @Override
    public JexlMethod getConstructor(Object obj, Object... args) {
      checkAccess(obj);
      return uberspect.getConstructor(obj, args);
    }

    @Override
    public JexlPropertyGet getPropertyGet(Object obj, Object identifier) {
      checkAccess(obj);
      return uberspect.getPropertyGet(obj, identifier);
    }

    @Override
    public JexlPropertyGet getPropertyGet(List<PropertyResolver> resolvers, Object obj, Object identifier) {
      checkAccess(obj);
      return uberspect.getPropertyGet(resolvers, obj, identifier);
    }

    @Override
    public JexlPropertySet getPropertySet(Object obj, Object identifier, Object arg) {
      checkAccess(obj);
      return uberspect.getPropertySet(obj, identifier, arg);
    }

    @Override
    public JexlPropertySet getPropertySet(List<PropertyResolver> resolvers, Object obj, Object identifier, Object arg) {
      checkAccess(obj);
      return uberspect.getPropertySet(resolvers, obj, identifier, arg);
    }

    @Override
    public Iterator<?> getIterator(Object obj) {
      checkAccess(obj);
      return uberspect.getIterator(obj);
    }

    @Override
    public JexlArithmetic.Uberspect getArithmetic(JexlArithmetic arithmetic) {
      return uberspect.getArithmetic(arithmetic);
    } 
  }
}
```

## References
* Apache Commons JEXL: [Project page](https://commons.apache.org/proper/commons-jexl/).
* Apache Commons JEXL documentation: [JEXL 2.1.1 API](https://commons.apache.org/proper/commons-jexl/javadocs/apidocs-2.1.1/).
* Apache Commons JEXL documentation: [JEXL 3.1 API](https://commons.apache.org/proper/commons-jexl/apidocs/index.html).
* OWASP: [Expression Language Injection](https://owasp.org/www-community/vulnerabilities/Expression_Language_Injection).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
