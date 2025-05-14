# Building a command with an injected environment variable
Passing unvalidated user input into the environment variables of a subprocess can allow an attacker to execute malicious code.


## Recommendation
If possible, use hard-coded string literals to specify the environment variable or its value. Instead of passing the user input directly to the process or library function, examine the user input and then choose among hard-coded string literals.

If the applicable environment variables cannot be determined at compile time, then add code to verify that the user input string is safe before using it.


## Example
In the following (BAD) example, the environment variable `PATH` is set to the value of the user input `path` without validation.


```java
public void doGet(HttpServletRequest request, HttpServletResponse response) {
    String path = request.getParameter("path");

    Map<String, String> env = processBuilder.environment();
    // BAD: path is tainted and being added to the environment
    env.put("PATH", path);

    processBuilder.start();
}
```
In the following (BAD) example, an environment variable is set with a name that is derived from the user input `var` without validation.


```java
public void doGet(HttpServletRequest request, HttpServletResponse response) {
    String attr = request.getParameter("attribute");
    String value = request.getParameter("value");

    Map<String, String> env = processBuilder.environment();
    // BAD: attr and value are tainted and being added to the environment
    env.put(attr, value);

    processBuilder.start();
}
```
In the following (GOOD) example, the user's input is validated before being used to set the environment variable.


```java
String opt = request.getParameter("opt");
String value = request.getParameter("value");

Map<String, String> env = processBuilder.environment();

// GOOD: opt and value are checked before being added to the environment
if (permittedJavaOptions.contains(opt) && validOption(opt, value)) {
    env.put(opt, value);
}
```
In the following (GOOD) example, the user's input is checked and used to determine an environment variable to add.


```java
Map<String, String> env = builder.environment();
String debug = request.getParameter("debug");

// GOOD: Checking the value and not tainting the variable added to the environment
if (debug != null) {
    env.put("PYTHONDEBUG", "1");
}

```

## References
* The Java Tutorials: [Environment Variables](https://docs.oracle.com/javase/tutorial/essential/environment/env.html).
* OWASP: [Command injection](https://owasp.org/www-community/attacks/Command_Injection).
* Common Weakness Enumeration: [CWE-78](https://cwe.mitre.org/data/definitions/78.html).
* Common Weakness Enumeration: [CWE-88](https://cwe.mitre.org/data/definitions/88.html).
* Common Weakness Enumeration: [CWE-454](https://cwe.mitre.org/data/definitions/454.html).
