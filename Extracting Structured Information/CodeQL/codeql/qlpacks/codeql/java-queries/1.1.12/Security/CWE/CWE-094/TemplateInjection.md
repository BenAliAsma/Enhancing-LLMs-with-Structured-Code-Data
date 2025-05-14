# Server-side template injection
Template injection occurs when user input is embedded in a template's code in an unsafe manner. An attacker can use native template syntax to inject a malicious payload into a template, which is then executed server-side. This permits the attacker to run arbitrary code in the server's context.


## Recommendation
To fix this, ensure that untrusted input is not used as part of a template's code. If the application requirements do not allow this, use a sandboxed environment where access to unsafe attributes and methods is prohibited.


## Example
In the example given below, an untrusted HTTP parameter `code` is used as a Velocity template string. This can lead to remote code execution.


```java
@Controller
public class VelocitySSTI {

	@GetMapping(value = "bad")
	public void bad(HttpServletRequest request) {
		Velocity.init();

		String code = request.getParameter("code");

		VelocityContext context = new VelocityContext();

		context.put("name", "Velocity");
		context.put("project", "Jakarta");

		StringWriter w = new StringWriter();
		// evaluate( Context context, Writer out, String logTag, String instring )
		Velocity.evaluate(context, w, "mystring", code);
	}
}

```
In the next example, the problem is avoided by using a fixed template string `s`. Since the template's code is not attacker-controlled in this case, this solution prevents the execution of untrusted code.


```java
@Controller
public class VelocitySSTI {

	@GetMapping(value = "good")
	public void good(HttpServletRequest request) {
		Velocity.init();
		VelocityContext context = new VelocityContext();

		context.put("name", "Velocity");
		context.put("project", "Jakarta");

		String s = "We are using $project $name to render this.";
		StringWriter w = new StringWriter();
		Velocity.evaluate(context, w, "mystring", s);
		System.out.println(" string : " + w);
	}
}

```

## References
* Portswigger: [Server Side Template Injection](https://portswigger.net/web-security/server-side-template-injection).
* Common Weakness Enumeration: [CWE-1336](https://cwe.mitre.org/data/definitions/1336.html).
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
