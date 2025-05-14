# Main Method in Java EE Web Components
Debug code can create unintended entry points in a deployed Java EE web application therefore should never make into production. There is no reason to have a main method in a Java EE web application. Having a main method in the Java EE application increases the attack surface that an attacker can exploit to attack the application logic.


## Recommendation
Remove the main method from web components including servlets, filters, and listeners.


## Example
The following example shows two ways of implementing web components. In the 'BAD' case, a main method is implemented. In the 'GOOD' case, no main method is implemented.


```java
public class WebComponentMain implements Servlet {
	// BAD - Implement a main method in servlet.
	public static void main(String[] args) throws Exception {
		// Connect to my server
		URL url = new URL("https://www.example.com");
		url.openConnection();
	}

	// GOOD - Not to have a main method in servlet.
}

```

## References
* Fortify: [J2EE Bad Practices: Leftover Debug Code](https://vulncat.fortify.com/en/detail?id=desc.structural.java.j2ee_badpractices_leftover_debug_code)
* SonarSource: [Web applications should not have a "main" method](https://rules.sonarsource.com/java/tag/owasp/RSPEC-2653)
* Carnegie Mellon University: [ENV06-J. Production code must not contain debugging entry points](https://wiki.sei.cmu.edu/confluence/display/java/ENV06-J.+Production+code+must+not+contain+debugging+entry+points)
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
