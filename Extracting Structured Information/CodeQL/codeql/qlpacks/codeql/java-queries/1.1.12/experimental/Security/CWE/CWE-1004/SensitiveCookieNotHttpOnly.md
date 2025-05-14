# Sensitive cookies without the HttpOnly response header set
Cross-Site Scripting (XSS) is categorized as one of the OWASP Top 10 Security Vulnerabilities. The `HttpOnly` flag directs compatible browsers to prevent client-side script from accessing cookies. Including the `HttpOnly` flag in the Set-Cookie HTTP response header for a sensitive cookie helps mitigate the risk associated with XSS where an attacker's script code attempts to read the contents of a cookie and exfiltrate information obtained.


## Recommendation
Use the `HttpOnly` flag when generating a cookie containing sensitive information to help mitigate the risk of client side script accessing the protected cookie.


## Example
The following example shows two ways of generating sensitive cookies. In the 'BAD' cases, the `HttpOnly` flag is not set. In the 'GOOD' cases, the `HttpOnly` flag is set.


```java
class SensitiveCookieNotHttpOnly {
    // GOOD - Create a sensitive cookie with the `HttpOnly` flag set.
    public void addCookie(String jwt_token, HttpServletRequest request, HttpServletResponse response) {
        Cookie jwtCookie =new Cookie("jwt_token", jwt_token);
        jwtCookie.setPath("/");
        jwtCookie.setMaxAge(3600*24*7);
        jwtCookie.setHttpOnly(true);
        response.addCookie(jwtCookie);
    }

    // BAD - Create a sensitive cookie without the `HttpOnly` flag set.
    public void addCookie2(String jwt_token, String userId, HttpServletRequest request, HttpServletResponse response) {
        Cookie jwtCookie =new Cookie("jwt_token", jwt_token);
        jwtCookie.setPath("/");
        jwtCookie.setMaxAge(3600*24*7);
        response.addCookie(jwtCookie);
    }

    // GOOD - Set a sensitive cookie header with the `HttpOnly` flag set.
    public void addCookie3(String authId, HttpServletRequest request, HttpServletResponse response) {
        response.addHeader("Set-Cookie", "token=" +authId + ";HttpOnly;Secure");
    }

    // BAD - Set a sensitive cookie header without the `HttpOnly` flag set.
    public void addCookie4(String authId, HttpServletRequest request, HttpServletResponse response) {
        response.addHeader("Set-Cookie", "token=" +authId + ";Secure");
    }
    
    // GOOD - Set a sensitive cookie header using the class `javax.ws.rs.core.Cookie` with the `HttpOnly` flag set through string concatenation.
    public void addCookie5(String accessKey, HttpServletRequest request, HttpServletResponse response) {
        response.setHeader("Set-Cookie", new NewCookie("session-access-key", accessKey, "/", null, null, 0, true) + ";HttpOnly");
    }

    // BAD - Set a sensitive cookie header using the class `javax.ws.rs.core.Cookie` without the `HttpOnly` flag set.
    public void addCookie6(String accessKey, HttpServletRequest request, HttpServletResponse response) {
        response.setHeader("Set-Cookie", new NewCookie("session-access-key", accessKey, "/", null, null, 0, true).toString());
    }

    // GOOD - Set a sensitive cookie header using the class `javax.ws.rs.core.Cookie` with the `HttpOnly` flag set through the constructor.
    public void addCookie7(String accessKey, HttpServletRequest request, HttpServletResponse response) {
        NewCookie accessKeyCookie = new NewCookie("session-access-key", accessKey, "/", null, null, 0, true, true);
        response.setHeader("Set-Cookie", accessKeyCookie.toString());
    }
}

```

## References
* PortSwigger: [Cookie without HttpOnly flag set](https://portswigger.net/kb/issues/00500600_cookie-without-httponly-flag-set)
* OWASP: [HttpOnly](https://owasp.org/www-community/HttpOnly)
* Common Weakness Enumeration: [CWE-1004](https://cwe.mitre.org/data/definitions/1004.html).
