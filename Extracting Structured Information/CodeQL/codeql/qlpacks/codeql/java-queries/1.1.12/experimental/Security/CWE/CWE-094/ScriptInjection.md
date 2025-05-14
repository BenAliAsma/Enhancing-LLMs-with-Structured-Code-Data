# Injection in Java Script Engine
The Java Scripting API has been available since the release of Java 6. It allows applications to interact with scripts written in languages such as JavaScript. It serves as an embedded scripting engine inside Java applications which allows Java-to-JavaScript interoperability and provides a seamless integration between the two languages. If an expression is built using attacker-controlled data, and then evaluated in a powerful context, it may allow the attacker to run arbitrary code.


## Recommendation
In general, including user input in a Java Script Engine expression should be avoided. If user input must be included in the expression, it should be then evaluated in a safe context that doesn't allow arbitrary code invocation. Use "Cloudbees Rhino Sandbox" or sandboxing with SecurityManager, which will be deprecated in a future release, or use [GraalVM](https://www.graalvm.org/) instead.


## Example
The following code could execute user-supplied JavaScript code in `ScriptEngine`


```java
// Bad: ScriptEngine allows arbitrary code injection
ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
ScriptEngine scriptEngine = scriptEngineManager.getEngineByExtension("js");
Object result = scriptEngine.eval(code);
```

```java
// Bad: Execute externally controlled input in Nashorn Script Engine
NashornScriptEngineFactory factory = new NashornScriptEngineFactory();
NashornScriptEngine engine = (NashornScriptEngine) factory.getScriptEngine(new String[] { "-scripting"});
Object result = engine.eval(input);

```
The following example shows two ways of using Rhino expression. In the 'BAD' case, an unsafe context is initialized with `initStandardObjects` that allows arbitrary Java code to be executed. In the 'GOOD' case, a safe context is initialized with `initSafeStandardObjects` or `setClassShutter`.


```java
import org.mozilla.javascript.ClassShutter;
import org.mozilla.javascript.Context;
import org.mozilla.javascript.Scriptable;

public class RhinoInjection extends HttpServlet {

  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    response.setContentType("text/plain");
    String code = request.getParameter("code");
    Context ctx = Context.enter();
    try {
      {
        // BAD: allow arbitrary Java and JavaScript code to be executed
        Scriptable scope = ctx.initStandardObjects();
      }

      {
        // GOOD: enable the safe mode
        Scriptable scope = ctx.initSafeStandardObjects();
      }

      {
        // GOOD: enforce a constraint on allowed classes
        Scriptable scope = ctx.initStandardObjects();
        ctx.setClassShutter(new ClassShutter() {
            public boolean visibleToScripts(String className) {
              return className.startsWith("com.example.");
            }
        });
      }

      Object result = ctx.evaluateString(scope, code, "<code>", 1, null);
      response.getWriter().print(Context.toString(result));
    } catch(RhinoException ex) {
      response.getWriter().println(ex.getMessage());
    } finally {
      Context.exit();
    }
  }
}

```

## References
* CERT coding standard: [ScriptEngine code injection](https://wiki.sei.cmu.edu/confluence/display/java/IDS52-J.+Prevent+code+injection)
* GraalVM: [Secure by Default](https://www.graalvm.org/reference-manual/js/NashornMigrationGuide/#secure-by-default)
* Mozilla Rhino: [Rhino: JavaScript in Java](https://github.com/mozilla/rhino)
* Rhino Sandbox: [A sandbox to execute JavaScript code with Rhino in Java](https://github.com/javadelight/delight-rhino-sandbox)
* GuardRails: [Code Injection](https://docs.guardrails.io/docs/en/vulnerabilities/java/insecure_use_of_dangerous_function#code-injection)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
