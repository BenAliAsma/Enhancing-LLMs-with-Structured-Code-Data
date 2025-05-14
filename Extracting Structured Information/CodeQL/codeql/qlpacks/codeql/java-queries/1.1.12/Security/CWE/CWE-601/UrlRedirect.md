# URL redirection from remote source
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but which is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.

If this is not possible, then the user input should be validated in some other way, for example, by verifying that the target URL is on the same host as the current page.


## Example
The following example shows an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks:


```java
public class UrlRedirect extends HttpServlet {
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    // BAD: a request parameter is incorporated without validation into a URL redirect
    response.sendRedirect(request.getParameter("target"));
  }
}
```
One way to remedy the problem is to validate the user input against a known fixed string before doing the redirection:


```java
public class UrlRedirect extends HttpServlet {
  private static final List<String> VALID_REDIRECTS = Arrays.asList(
    "http://cwe.mitre.org/data/definitions/601.html",
    "http://cwe.mitre.org/data/definitions/79.html"
  );

  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    // GOOD: the request parameter is validated against a known list of strings
    String target = request.getParameter("target");
    if (VALID_REDIRECTS.contains(target)) {
        response.sendRedirect(target);
    } else {
        response.sendRedirect("/error.html");
    }
  }
}
```
Alternatively, we can check that the target URL does not redirect to a different host by checking that the URL is either relative or on a known good host:


```java
public class UrlRedirect extends HttpServlet {
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    try {
      String urlString = request.getParameter("page");
      URI url = new URI(urlString);

      if (!url.isAbsolute()) {
        response.sendRedirect(url.toString()); // GOOD: The redirect is to a relative URL
      }

      if ("example.org".equals(url.getHost())) {
        response.sendRedirect(url.toString()); // GOOD: The redirect is to a known host
      }
    } catch (URISyntaxException e) {
        // handle exception
    }
  }
}
```
Note that as written, the above code will allow redirects to URLs on `example.com`, which is harmless but perhaps not intended. You can substitute your own domain (if known) for `example.com` to prevent this.


## References
* OWASP: [ Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Microsoft Docs: [Preventing Open Redirection Attacks (C\#)](https://docs.microsoft.com/en-us/aspnet/mvc/overview/security/preventing-open-redirection-attacks).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).
