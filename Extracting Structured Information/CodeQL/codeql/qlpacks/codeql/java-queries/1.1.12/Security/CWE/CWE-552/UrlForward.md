# URL forward from a remote source
Directly incorporating user input into a URL forward request without validating the input can cause file information disclosure by allowing an attacker to access unauthorized URLs.


## Recommendation
To guard against untrusted URL forwarding, you should avoid putting user input directly into a forwarded URL. Instead, you should maintain a list of authorized URLs on the server, then choose from that list based on the user input provided.


## Example
The following example shows an HTTP request parameter being used directly in a URL forward without validating the input, which may cause file information disclosure. It also shows how to remedy the problem by validating the user input against a known fixed string.


```java
public class UrlForward extends HttpServlet {
	private static final String VALID_FORWARD = "https://cwe.mitre.org/data/definitions/552.html";

	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		ServletConfig cfg = getServletConfig();
		ServletContext sc = cfg.getServletContext();

		// BAD: a request parameter is incorporated without validation into a URL forward
		sc.getRequestDispatcher(request.getParameter("target")).forward(request, response);

		// GOOD: the request parameter is validated against a known fixed string
		if (VALID_FORWARD.equals(request.getParameter("target"))) {
			sc.getRequestDispatcher(VALID_FORWARD).forward(request, response);
		}
	}
}

```

## References
* OWASP: [Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-552](https://cwe.mitre.org/data/definitions/552.html).
