# JShell injection
The Java Shell tool (JShell) is an interactive tool for learning the Java programming language and prototyping Java code. JShell is a Read-Evaluate-Print Loop (REPL), which evaluates declarations, statements, and expressions as they are entered and immediately shows the results. If an expression is built using attacker-controlled data and then evaluated, it may allow the attacker to run arbitrary code.


## Recommendation
It is generally recommended to avoid using untrusted input in a JShell expression. If it is not possible, JShell expressions should be run in a sandbox that allows accessing only explicitly allowed classes.


## Example
The following example calls `JShell.eval(...)` or `SourceCodeAnalysis.wrappers(...)` to execute untrusted data.


```java
import javax.servlet.http.HttpServletRequest;
import jdk.jshell.JShell;
import jdk.jshell.SourceCodeAnalysis;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class JShellInjection {

	@GetMapping(value = "bad1")
	public void bad1(HttpServletRequest request) {
		String input = request.getParameter("code");
		JShell jShell = JShell.builder().build();
        // BAD: allow execution of arbitrary Java code
		jShell.eval(input);
	}

	@GetMapping(value = "bad2")
	public void bad2(HttpServletRequest request) {
		String input = request.getParameter("code");
		JShell jShell = JShell.builder().build();
		SourceCodeAnalysis sourceCodeAnalysis = jShell.sourceCodeAnalysis();
        // BAD: allow execution of arbitrary Java code
		sourceCodeAnalysis.wrappers(input);
	}

	@GetMapping(value = "bad3")
	public void bad3(HttpServletRequest request) {
		String input = request.getParameter("code");
		JShell jShell = JShell.builder().build();
		SourceCodeAnalysis.CompletionInfo info;
		SourceCodeAnalysis sca = jShell.sourceCodeAnalysis();
		for (info = sca.analyzeCompletion(input);
				info.completeness().isComplete();
				info = sca.analyzeCompletion(info.remaining())) {
			// BAD: allow execution of arbitrary Java code
			jShell.eval(info.source());
		}
	}
}
```

## References
* Java Shell User’s Guide: [Introduction to JShell](https://docs.oracle.com/en/java/javase/11/jshell/introduction-jshell.html)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
