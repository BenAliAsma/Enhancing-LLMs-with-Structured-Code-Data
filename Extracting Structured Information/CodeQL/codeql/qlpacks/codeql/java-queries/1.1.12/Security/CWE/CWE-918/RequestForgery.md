# Server-side request forgery
Directly incorporating user input into an HTTP request without validating the input can facilitate server-side request forgery (SSRF) attacks. In these attacks, the server may be tricked into making a request and interacting with an attacker-controlled server.


## Recommendation
To guard against SSRF attacks, you should avoid putting user-provided input directly into a request URL. Instead, maintain a list of authorized URLs on the server; then choose from that list based on the input provided. Alternatively, ensure requests constructed from user input are limited to a particular host or more restrictive URL prefix.


## Example
The following example shows an HTTP request parameter being used directly to form a new request without validating the input, which facilitates SSRF attacks. It also shows how to remedy the problem by validating the user input against a known fixed string.


```java
import java.net.http.HttpClient;

public class SSRF extends HttpServlet {
	private static final String VALID_URI = "http://lgtm.com";
	private HttpClient client = HttpClient.newHttpClient();

	protected void doGet(HttpServletRequest request, HttpServletResponse response)
		throws ServletException, IOException {
		URI uri = new URI(request.getParameter("uri"));
		// BAD: a request parameter is incorporated without validation into a Http request
		HttpRequest r = HttpRequest.newBuilder(uri).build();
		client.send(r, null);

		// GOOD: the request parameter is validated against a known fixed string
		if (VALID_URI.equals(request.getParameter("uri"))) {
			HttpRequest r2 = HttpRequest.newBuilder(uri).build();
			client.send(r2, null);
		}
	}
}

```

## References
* [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
* Common Weakness Enumeration: [CWE-918](https://cwe.mitre.org/data/definitions/918.html).
