# Injection in Jython
Python has been the most widely used programming language in recent years, and Jython (formerly known as JPython) is a popular Java implementation of Python. It allows embedded Python scripting inside Java applications and provides an interactive interpreter that can be used to interact with Java packages or with running Java applications. If an expression is built using attacker-controlled data and then evaluated, it may allow the attacker to run arbitrary code.


## Recommendation
In general, including user input in Jython expression should be avoided. If user input must be included in an expression, it should be then evaluated in a safe context that doesn't allow arbitrary code invocation.


## Example
The following code could execute arbitrary code in Jython Interpreter


```java
import org.python.util.PythonInterpreter;

public class JythonInjection extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/plain");
        String code = request.getParameter("code");
        PythonInterpreter interpreter = null;
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        try {
            interpreter = new PythonInterpreter();
            interpreter.setOut(out);
            interpreter.setErr(out);

            // BAD: allow execution of arbitrary Python code
            interpreter.exec(code);
            out.flush();

            response.getWriter().print(out.toString());
        } catch(PyException ex) {
            response.getWriter().println(ex.getMessage());
        } finally {
            if (interpreter != null) {
                interpreter.close();
            }
            out.close();
        }
    }

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/plain");
        String code = request.getParameter("code");
        PythonInterpreter interpreter = null;

        try {
            interpreter = new PythonInterpreter();
            // BAD: allow execution of arbitrary Python code
            PyObject py = interpreter.eval(code);

            response.getWriter().print(py.toString());
        } catch(PyException ex) {
            response.getWriter().println(ex.getMessage());
        } finally {
            if (interpreter != null) {
                interpreter.close();
            }
        }
    }
}

```

## References
* Jython Organization: [Jython and Java Integration](https://jython.readthedocs.io/en/latest/JythonAndJavaIntegration/)
* PortSwigger: [Python code injection](https://portswigger.net/kb/issues/00100f10_python-code-injection)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
* Common Weakness Enumeration: [CWE-95](https://cwe.mitre.org/data/definitions/95.html).
