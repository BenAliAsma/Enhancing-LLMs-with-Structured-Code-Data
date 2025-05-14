# Disabled Spring CSRF protection
When you set up a web server to receive a request from a client without any mechanism for verifying that it was intentionally sent, then it is vulnerable to attack. An attacker can trick a client into making an unintended request to the web server that will be treated as an authentic request. This can be done via a URL, image load, XMLHttpRequest, etc. and can result in exposure of data or unintended code execution.


## Recommendation
When you use Spring, Cross-Site Request Forgery (CSRF) protection is enabled by default. Spring's recommendation is to use CSRF protection for any request that could be processed by a browser client by normal users.


## Example
The following example shows the Spring Java configuration with CSRF protection disabled. This type of configuration should only be used if you are creating a service that is used only by non-browser clients.


```java
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@EnableWebSecurity
@Configuration
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
  @Override
  protected void configure(HttpSecurity http) throws Exception {
    http
      .csrf(csrf ->
        // BAD - CSRF protection shouldn't be disabled
        csrf.disable() 
      );
  }
}

```

## References
* OWASP: [Cross-Site Request Forgery (CSRF)](https://www.owasp.org/index.php/Cross-Site_Request_Forgery_(CSRF)).
* Spring Security Reference: [ Cross Site Request Forgery (CSRF) ](https://docs.spring.io/spring-security/reference/servlet/exploits/csrf.html).
* Common Weakness Enumeration: [CWE-352](https://cwe.mitre.org/data/definitions/352.html).
