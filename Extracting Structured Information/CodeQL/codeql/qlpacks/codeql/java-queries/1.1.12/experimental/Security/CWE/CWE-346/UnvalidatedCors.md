# CORS is derived from untrusted input
A server can send the `Access-Control-Allow-Credentials` CORS header to control when a browser may send user credentials in Cross-Origin HTTP requests.

When the `Access-Control-Allow-Credentials` header is `true`, the `Access-Control-Allow-Origin` header must have a value different from `*` in order for browsers to accept the header. Therefore, to allow multiple origins for cross-origin requests with credentials, the server must dynamically compute the value of the `Access-Control-Allow-Origin` header. Computing this header value from information in the request to the server can therefore potentially allow an attacker to control the origins that the browser sends credentials to.


## Recommendation
When the `Access-Control-Allow-Credentials` header value is `true`, a dynamic computation of the `Access-Control-Allow-Origin` header must involve sanitization if it relies on user-controlled input.

Since the `null` origin is easy to obtain for an attacker, it is never safe to use `null` as the value of the `Access-Control-Allow-Origin` header when the `Access-Control-Allow-Credentials` header value is `true`.A null origin can be set by an attacker using a sandboxed iframe. A more detailed explanation is available in the portswigger blogpost referenced below.


## Example
In the example below, the server allows the browser to send user credentials in a cross-origin request. The request header `origins` controls the allowed origins for such a Cross-Origin request.


```java
import java.io.IOException;

import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.lang3.StringUtils;

public class CorsFilter implements Filter {
    public void init(FilterConfig filterConfig) throws ServletException {}

    public void doFilter(ServletRequest req, ServletResponse res,
            FilterChain chain) throws IOException, ServletException {
        HttpServletRequest request = (HttpServletRequest) req;
        HttpServletResponse response = (HttpServletResponse) res;
        String url = request.getHeader("Origin");

        if (!StringUtils.isEmpty(url)) {
            String val = response.getHeader("Access-Control-Allow-Origin");

            if (StringUtils.isEmpty(val)) {
                response.addHeader("Access-Control-Allow-Origin", url); // BAD -> User controlled CORS header being set here.
                response.addHeader("Access-Control-Allow-Credentials", "true");
            }
        }

        if (!StringUtils.isEmpty(url)) {
            List<String> checkorigins = Arrays.asList("www.example.com", "www.sub.example.com");

            if (checkorigins.contains(url)) { // GOOD -> Origin is validated here.
                response.addHeader("Access-Control-Allow-Origin", url);
                response.addHeader("Access-Control-Allow-Credentials", "true");
            }
        }

        chain.doFilter(req, res);
    }

    public void destroy() {}
}

```

## References
* Mozilla Developer Network: [CORS, Access-Control-Allow-Origin](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin).
* Mozilla Developer Network: [CORS, Access-Control-Allow-Credentials](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials).
* PortSwigger: [Exploiting CORS Misconfigurations for Bitcoins and Bounties](http://blog.portswigger.net/2016/10/exploiting-cors-misconfigurations-for.html)
* W3C: [CORS for developers, Advice for Resource Owners](https://w3c.github.io/webappsec-cors-for-developers/#resources)
* Common Weakness Enumeration: [CWE-346](https://cwe.mitre.org/data/definitions/346.html).
