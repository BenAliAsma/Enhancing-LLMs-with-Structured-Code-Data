# Exposed Spring Boot actuators
Spring Boot includes a number of additional features called actuators that let you monitor and interact with your web application. Exposing unprotected actuator endpoints via JXM or HTTP can, however, lead to information disclosure or even to remote code execution vulnerability.


## Recommendation
Since actuator endpoints may contain sensitive information, careful consideration should be given about when to expose them. You should take care to secure exposed HTTP endpoints in the same way that you would any other sensitive URL. If Spring Security is present, endpoints are secured by default using Spring Security’s content-negotiation strategy. If you wish to configure custom security for HTTP endpoints, for example, only allow users with a certain role to access them, Spring Boot provides some convenient `RequestMatcher` objects that can be used in combination with Spring Security.


## Example
In the first example, the custom security configuration allows unauthenticated access to all actuator endpoints. This may lead to sensitive information disclosure and should be avoided.

In the second example, only users with `ENDPOINT_ADMIN` role are allowed to access the actuator endpoints.


```java
@Configuration(proxyBeanMethods = false)
public class SpringBootActuators extends WebSecurityConfigurerAdapter {

  @Override
  protected void configure(HttpSecurity http) throws Exception {
    // BAD: Unauthenticated access to Spring Boot actuator endpoints is allowed
    http.requestMatcher(EndpointRequest.toAnyEndpoint()).authorizeRequests((requests) ->
        requests.anyRequest().permitAll());
  }
}

@Configuration(proxyBeanMethods = false)
public class ActuatorSecurity extends WebSecurityConfigurerAdapter {

  @Override
  protected void configure(HttpSecurity http) throws Exception {
    // GOOD: only users with ENDPOINT_ADMIN role are allowed to access the actuator endpoints
    http.requestMatcher(EndpointRequest.toAnyEndpoint()).authorizeRequests((requests) ->
        requests.anyRequest().hasRole("ENDPOINT_ADMIN"));
    http.httpBasic();
  }
}
```

## References
* Spring Boot documentation: [Actuators](https://docs.spring.io/spring-boot/docs/current/reference/html/production-ready-features.html).
* [Exploiting Spring Boot Actuators](https://www.veracode.com/blog/research/exploiting-spring-boot-actuators)
* Common Weakness Enumeration: [CWE-16](https://cwe.mitre.org/data/definitions/16.html).
