# Groovy Language injection
Apache Groovy is a powerful, optionally typed and dynamic language, with static-typing and static compilation capabilities. It integrates smoothly with any Java program, and immediately delivers to your application powerful features, including scripting capabilities, Domain-Specific Language authoring, runtime and compile-time meta-programming and functional programming. If a Groovy script is built using attacker-controlled data, and then evaluated, then it may allow the attacker to achieve RCE.


## Recommendation
It is generally recommended to avoid using untrusted input in a Groovy evaluation. If this is not possible, use a sandbox solution. Developers must also take care that Groovy compile-time metaprogramming can also lead to RCE: it is possible to achieve RCE by compiling a Groovy script (see the article "Abusing Meta Programming for Unauthenticated RCE!" linked below). Groovy's `SecureASTCustomizer` allows securing source code by controlling what code constructs are permitted. This is typically done when using Groovy for its scripting or domain specific language (DSL) features. The fundamental problem is that Groovy is a dynamic language, yet `SecureASTCustomizer` works by looking at Groovy AST statically. This makes it very easy for an attacker to bypass many of the intended checks (see \[Groovy SecureASTCustomizer is harmful\](https://kohsuke.org/2012/04/27/groovy-secureastcustomizer-is-harmful/)). Therefore, besides `SecureASTCustomizer`, runtime checks are also necessary before calling Groovy methods (see \[Improved sandboxing of Groovy scripts\](https://melix.github.io/blog/2015/03/sandboxing.html)). It is also possible to use a block-list method, excluding unwanted classes from being loaded by the JVM. This method is not always recommended, because block-lists can be bypassed by unexpected values.


## Example
The following example uses untrusted data to evaluate a Groovy script.


```java
public class GroovyInjection {
    void injectionViaClassLoader(HttpServletRequest request) {    
        String script = request.getParameter("script");
        final GroovyClassLoader classLoader = new GroovyClassLoader();
        Class groovy = classLoader.parseClass(script);
        GroovyObject groovyObj = (GroovyObject) groovy.newInstance();
    }

    void injectionViaEval(HttpServletRequest request) {
        String script = request.getParameter("script");
        Eval.me(script);
    }

    void injectionViaGroovyShell(HttpServletRequest request) {
        GroovyShell shell = new GroovyShell();
        String script = request.getParameter("script");
        shell.evaluate(script);
    }

    void injectionViaGroovyShellGroovyCodeSource(HttpServletRequest request) {
        GroovyShell shell = new GroovyShell();
        String script = request.getParameter("script");
        GroovyCodeSource gcs = new GroovyCodeSource(script, "test", "Test");
        shell.evaluate(gcs);
    }
}


```
The following example uses classloader block-list approach to exclude loading dangerous classes.


```java
public class SandboxGroovyClassLoader extends ClassLoader {
    public SandboxGroovyClassLoader(ClassLoader parent) {
        super(parent);
    }

    /* override `loadClass` here to prevent loading sensitive classes, such as `java.lang.Runtime`, `java.lang.ProcessBuilder`, `java.lang.System`, etc.  */
    /* Note we must also block `groovy.transform.ASTTest`, `groovy.lang.GrabConfig` and `org.buildobjects.process.ProcBuilder` to prevent compile-time RCE. */

    static void runWithSandboxGroovyClassLoader() throws Exception {
        // GOOD: route all class-loading via sand-boxing classloader.
        SandboxGroovyClassLoader classLoader = new GroovyClassLoader(new SandboxGroovyClassLoader());
        
        Class<?> scriptClass = classLoader.parseClass(untrusted.getQueryString());
        Object scriptInstance = scriptClass.newInstance();
        Object result = scriptClass.getDeclaredMethod("bar", new Class[]{}).invoke(scriptInstance, new Object[]{});
    }
}
```

## References
* Orange Tsai: [Abusing Meta Programming for Unauthenticated RCE!](https://blog.orange.tw/2019/02/abusing-meta-programming-for-unauthenticated-rce.html).
* Cédric Champeau: [Improved sandboxing of Groovy scripts](https://melix.github.io/blog/2015/03/sandboxing.html).
* Kohsuke Kawaguchi: [Groovy SecureASTCustomizer is harmful](https://kohsuke.org/2012/04/27/groovy-secureastcustomizer-is-harmful/).
* Welk1n: [Groovy Injection payloads](https://github.com/welk1n/exploiting-groovy-in-Java/).
* Charles Chan: [Secure Groovy Script Execution in a Sandbox](https://levelup.gitconnected.com/secure-groovy-script-execution-in-a-sandbox-ea39f80ee87/).
* Eugene: [Scripting and sandboxing in a JVM environment](https://stringconcat.com/en/scripting-and-sandboxing/).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
