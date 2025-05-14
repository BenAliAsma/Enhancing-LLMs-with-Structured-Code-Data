# IP address spoofing
An original client IP address is retrieved from an http header (`X-Forwarded-For` or `X-Real-IP` or `Proxy-Client-IP` etc.), which is used to ensure security. Attackers can forge the value of these identifiers to bypass a ban-list, for example.


## Recommendation
Do not trust the values of HTTP headers allegedly identifying the originating IP. If you are aware your application will run behind some reverse proxies then the last entry of a `X-Forwarded-For` header value may be more trustworthy than the rest of it because some reverse proxies append the IP address they observed to the end of any remote-supplied header.


## Example
The following examples show the bad case and the good case respectively. In `bad1` method and `bad2` method, the client ip the `X-Forwarded-For` is split into comma-separated values, but the less-trustworthy first one is used. Both of these examples could be deceived by providing a forged HTTP header. The method `good1` similarly splits an `X-Forwarded-For` value, but uses the last, more-trustworthy entry.


```java
import javax.servlet.http.HttpServletRequest;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class ClientSuppliedIpUsedInSecurityCheck {

    @Autowired
    private HttpServletRequest request;

    @GetMapping(value = "bad1")
    public void bad1(HttpServletRequest request) {
        String ip = getClientIP();
        if (!StringUtils.startsWith(ip, "192.168.")) {
            new Exception("ip illegal");
        }
    }

    @GetMapping(value = "bad2")
    public void bad2(HttpServletRequest request) {
        String ip = getClientIP();
        if (!"127.0.0.1".equals(ip)) {
            new Exception("ip illegal");
        }
    }

    @GetMapping(value = "good1")
    @ResponseBody
    public String good1(HttpServletRequest request) {
        String ip = request.getHeader("X-FORWARDED-FOR");
        // Good: if this application runs behind a reverse proxy it may append the real remote IP to the end of any client-supplied X-Forwarded-For header.
        ip = ip.split(",")[ip.split(",").length - 1];
        if (!StringUtils.startsWith(ip, "192.168.")) {
            new Exception("ip illegal");
        }
        return ip;
    }

    protected String getClientIP() {
        String xfHeader = request.getHeader("X-Forwarded-For");
        if (xfHeader == null) {
            return request.getRemoteAddr();
        }
        return xfHeader.split(",")[0];
    }
}

```

## References
* Dennis Schneider: [ Prevent IP address spoofing with X-Forwarded-For header when using AWS ELB and Clojure Ring](https://www.dennis-schneider.com/blog/prevent-ip-address-spoofing-with-x-forwarded-for-header-and-aws-elb-in-clojure-ring/)
* Security Rule Zero: [A Warning about X-Forwarded-For](https://www.f5.com/company/blog/security-rule-zero-a-warning-about-x-forwarded-for)
* Common Weakness Enumeration: [CWE-348](https://cwe.mitre.org/data/definitions/348.html).
