# JNDI lookup with user-controlled name
The Java Naming and Directory Interface (JNDI) is a Java API for a directory service that allows Java software clients to discover and look up data and resources (in the form of Java objects) via a name. If the name being used to look up the data is controlled by the user, it can point to a malicious server, which can return an arbitrary object. In the worst case, this can allow remote code execution.


## Recommendation
The general recommendation is to avoid passing untrusted data to the `InitialContext.lookup ` method. If the name being used to look up the object must be provided by the user, make sure that it's not in the form of an absolute URL or that it's the URL pointing to a trusted server.


## Example
In the following examples, the code accepts a name from the user, which it uses to look up an object.

In the first example, the user provided name is used to look up an object.

The second example validates the name before using it to look up an object.


```java
import javax.naming.Context;
import javax.naming.InitialContext;

public void jndiLookup(HttpServletRequest request) throws NamingException {
  String name = request.getParameter("name");

  Hashtable<String, String> env = new Hashtable<String, String>();
  env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.rmi.registry.RegistryContextFactory");
  env.put(Context.PROVIDER_URL, "rmi://trusted-server:1099");
  InitialContext ctx = new InitialContext(env);

  // BAD: User input used in lookup
  ctx.lookup(name);

  // GOOD: The name is validated before being used in lookup
  if (isValid(name)) {
    ctx.lookup(name);
  } else {
    // Reject the request
  }
}
```

## References
* Oracle: [Java Naming and Directory Interface (JNDI)](https://docs.oracle.com/javase/8/docs/technotes/guides/jndi/).
* Black Hat materials: [A Journey from JNDI/LDAP Manipulation to Remote Code Execution Dream Land](https://www.blackhat.com/docs/us-16/materials/us-16-Munoz-A-Journey-From-JNDI-LDAP-Manipulation-To-RCE-wp.pdf).
* Veracode: [Exploiting JNDI Injections in Java](https://www.veracode.com/blog/research/exploiting-jndi-injections-java).
* Common Weakness Enumeration: [CWE-74](https://cwe.mitre.org/data/definitions/74.html).
