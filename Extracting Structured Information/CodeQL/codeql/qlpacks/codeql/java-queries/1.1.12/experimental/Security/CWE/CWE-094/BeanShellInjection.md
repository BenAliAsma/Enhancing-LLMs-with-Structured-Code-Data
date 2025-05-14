# BeanShell injection
BeanShell is a small, free, embeddable Java source interpreter with object scripting language features, written in Java. BeanShell dynamically executes standard Java syntax and extends it with common scripting conveniences such as loose types, commands, and method closures like those in Perl and JavaScript. If a BeanShell expression is built using attacker-controlled data, and then evaluated, then it may allow the attacker to run arbitrary code.


## Recommendation
It is generally recommended to avoid using untrusted input in a BeanShell expression. If it is not possible, BeanShell expressions should be run in a sandbox that allows accessing only explicitly allowed classes.


## Example
The following example uses untrusted data to build and run a BeanShell expression.


```java
import bsh.Interpreter;
import javax.servlet.http.HttpServletRequest;
import org.springframework.scripting.bsh.BshScriptEvaluator;
import org.springframework.scripting.support.StaticScriptSource;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class BeanShellInjection {

	@GetMapping(value = "bad1")
	public void bad1(HttpServletRequest request) {
		String code = request.getParameter("code");
		BshScriptEvaluator evaluator = new BshScriptEvaluator();
		evaluator.evaluate(new StaticScriptSource(code)); //bad
	}

	@GetMapping(value = "bad2")
	public void bad2(HttpServletRequest request) throws Exception {
		String code = request.getParameter("code");
		Interpreter interpreter = new Interpreter();
		interpreter.eval(code);  //bad
	}

	@GetMapping(value = "bad3")
	public void bad3(HttpServletRequest request) {
		String code = request.getParameter("code");
		StaticScriptSource staticScriptSource = new StaticScriptSource("test");
		staticScriptSource.setScript(code);
		BshScriptEvaluator evaluator = new BshScriptEvaluator();
		evaluator.evaluate(staticScriptSource);  //bad
	}
}

```

## References
* CVE-2016-2510:[BeanShell Injection](https://nvd.nist.gov/vuln/detail/CVE-2016-2510).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
