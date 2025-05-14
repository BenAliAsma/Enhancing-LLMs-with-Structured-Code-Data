# Regular expression injection
Constructing a regular expression with unsanitized user input is dangerous as a malicious user may be able to modify the meaning of the expression. In particular, such a user may be able to provide a regular expression fragment that takes exponential time in the worst case, and use that to perform a Denial of Service attack.


## Recommendation
Before embedding user input into a regular expression, use a sanitization function such as `Pattern.quote` to escape meta-characters that have special meaning.


## Example
The following example shows an HTTP request parameter that is used to construct a regular expression.

In the first case the user-provided regex is not escaped. If a malicious user provides a regex whose worst-case performance is exponential, then this could lead to a Denial of Service.

In the second case, the user input is escaped using `Pattern.quote` before being included in the regular expression. This ensures that the user cannot insert characters which have a special meaning in regular expressions.


```java
import java.util.regex.Pattern;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;

public class RegexInjectionDemo extends HttpServlet {

  public boolean badExample(javax.servlet.http.HttpServletRequest request) {
    String regex = request.getParameter("regex");
    String input = request.getParameter("input");

    // BAD: Unsanitized user input is used to construct a regular expression
    return input.matches(regex);
  }

  public boolean goodExample(javax.servlet.http.HttpServletRequest request) {
    String regex = request.getParameter("regex");
    String input = request.getParameter("input");

    // GOOD: User input is sanitized before constructing the regex
    return input.matches(Pattern.quote(regex));
  }
}

```

## References
* OWASP: [Regular expression Denial of Service - ReDoS](https://www.owasp.org/index.php/Regular_expression_Denial_of_Service_-_ReDoS).
* Wikipedia: [ReDoS](https://en.wikipedia.org/wiki/ReDoS).
* Java API Specification: [Pattern.quote](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/regex/Pattern.html#quote(java.lang.String)).
* Common Weakness Enumeration: [CWE-730](https://cwe.mitre.org/data/definitions/730.html).
* Common Weakness Enumeration: [CWE-400](https://cwe.mitre.org/data/definitions/400.html).
