# Disabled Netty HTTP header validation
Directly writing user input (for example, an HTTP request parameter) to an HTTP header can lead to an HTTP request-splitting or response-splitting vulnerability.

HTTP response splitting can lead to vulnerabilities such as XSS and cache poisoning.

HTTP request splitting can allow an attacker to inject an additional HTTP request into a client's outgoing socket connection. This can allow an attacker to perform an SSRF-like attack.

In the context of a servlet container, if the user input includes blank lines and the servlet container does not escape the blank lines, then a remote user can cause the response to turn into two separate responses. The remote user can then control one or more responses, which is also HTTP response splitting.


## Recommendation
Guard against HTTP header splitting in the same way as guarding against cross-site scripting. Before passing any data into HTTP headers, either check the data for special characters, or escape any special characters that are present.

If the code calls Netty API's directly, ensure that the `validateHeaders` parameter is set to `true`.


## Example
The following example shows the 'name' parameter being written to a cookie in two different ways. The first way writes it directly to the cookie, and thus is vulnerable to response-splitting attacks. The second way first removes all special characters, thus avoiding the potential problem.


```java
public class ResponseSplitting extends HttpServlet {
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
	throws ServletException, IOException {
		// BAD: setting a cookie with an unvalidated parameter
		Cookie cookie = new Cookie("name", request.getParameter("name"));
		response.addCookie(cookie);

		// GOOD: remove special characters before putting them in the header
		String name = removeSpecial(request.getParameter("name"));
		Cookie cookie2 = new Cookie("name", name);
		response.addCookie(cookie2);
	}

	private static String removeSpecial(String str) {
		return str.replaceAll("[^a-zA-Z ]", "");
	}
}

```

## Example
The following example shows the use of the library 'netty' with HTTP response-splitting verification configurations. The second way will verify the parameters before using them to build the HTTP response.


```java
import io.netty.handler.codec.http.DefaultHttpHeaders;

public class ResponseSplitting {
    // BAD: Disables the internal response splitting verification
    private final DefaultHttpHeaders badHeaders = new DefaultHttpHeaders(false);

    // GOOD: Verifies headers passed don't contain CRLF characters
    private final DefaultHttpHeaders goodHeaders = new DefaultHttpHeaders();

    // BAD: Disables the internal response splitting verification
    private final DefaultHttpResponse badResponse = new DefaultHttpResponse(version, httpResponseStatus, false);

    // GOOD: Verifies headers passed don't contain CRLF characters
    private final DefaultHttpResponse goodResponse = new DefaultHttpResponse(version, httpResponseStatus);
}

```

## Example
The following example shows the use of the netty library with configurations for verification of HTTP request splitting. The second recommended approach in the example verifies the parameters before using them to build the HTTP request.


```java
public class NettyRequestSplitting {
    // BAD: Disables the internal request splitting verification
    private final DefaultHttpHeaders badHeaders = new DefaultHttpHeaders(false);

    // GOOD: Verifies headers passed don't contain CRLF characters
    private final DefaultHttpHeaders goodHeaders = new DefaultHttpHeaders();

    // BAD: Disables the internal request splitting verification
    private final DefaultHttpRequest badRequest = new DefaultHttpRequest(httpVersion, method, uri, false);

    // GOOD: Verifies headers passed don't contain CRLF characters
    private final DefaultHttpRequest goodResponse = new DefaultHttpRequest(httpVersion, method, uri);
}

```

## References
* SecLists.org: [HTTP response splitting](https://seclists.org/bugtraq/2005/Apr/187).
* OWASP: [HTTP Response Splitting](https://www.owasp.org/index.php/HTTP_Response_Splitting).
* Wikipedia: [HTTP response splitting](http://en.wikipedia.org/wiki/HTTP_response_splitting).
* CAPEC: [CAPEC-105: HTTP Request Splitting](https://capec.mitre.org/data/definitions/105.html)
* Common Weakness Enumeration: [CWE-93](https://cwe.mitre.org/data/definitions/93.html).
* Common Weakness Enumeration: [CWE-113](https://cwe.mitre.org/data/definitions/113.html).
