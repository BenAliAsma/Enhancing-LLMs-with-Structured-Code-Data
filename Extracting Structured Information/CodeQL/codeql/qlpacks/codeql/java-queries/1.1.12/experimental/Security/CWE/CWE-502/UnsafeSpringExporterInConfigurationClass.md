# Unsafe deserialization with Spring's remote service exporters.
The Spring Framework provides several classes for creating remote service exporters. Under the hood, the exporters use various deserialization mechanisms such as `ObjectInputStream` or Hessian. Deserializing untrusted data is easily exploitable and in many cases allows an attacker to execute arbitrary code. If a remote attacker can reach endpoints created by the exporters, it results in remote code execution in the worst case.

Examples of unsafe exporters include: `HttpInvokerServiceExporter`, `SimpleHttpInvokerServiceExporter`, `RmiServiceExporter`, `HessianServiceExporter`.

CVE-2016-1000027 has been assigned to this issue in the Spring Framework. It is regarded as a design limitation, and can be mitigated but not fixed outright.


## Recommendation
Avoid using unsafe service exporters. Instead, use other message formats for API endpoints (for example, JSON), but make sure that the underlying deserialization mechanism is properly configured so that deserialization attacks are not possible. If the vulnerable exporters can not be replaced, consider using global deserialization filters introduced in JEP 290.


## Example
The following example shows how a vulnerable HTTP endpoint can be defined using `HttpInvokerServiceExporter` and Spring annotations:


```java
@Configuration
public class Server {

    @Bean(name = "/account")
    HttpInvokerServiceExporter accountService() {
        HttpInvokerServiceExporter exporter = new HttpInvokerServiceExporter();
        exporter.setService(new AccountServiceImpl());
        exporter.setServiceInterface(AccountService.class);
        return exporter;
    }

}

class AccountServiceImpl implements AccountService {

    @Override
    public String echo(String data) {
        return data;
    }
}

interface AccountService {
    String echo(String data);
}
```

## References
* OWASP: [Deserialization of untrusted data](https://www.owasp.org/index.php/Deserialization_of_untrusted_data).
* Spring Framework API documentation: [RemoteInvocationSerializingExporter class](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/remoting/rmi/RemoteInvocationSerializingExporter.html)
* Spring Framework API documentation: [HttpInvokerServiceExporter class](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/remoting/httpinvoker/HttpInvokerServiceExporter.html)
* National Vulnerability Database: [CVE-2016-1000027](https://nvd.nist.gov/vuln/detail/CVE-2016-1000027)
* Tenable Research Advisory: [\[R2\] Pivotal Spring Framework HttpInvokerServiceExporter readRemoteInvocation Method Untrusted Java Deserialization](https://www.tenable.com/security/research/tra-2016-20)
* Spring Framework bug tracker: [Sonatype vulnerability CVE-2016-1000027 in Spring-web project](https://github.com/spring-projects/spring-framework/issues/24434)
* OpenJDK: [JEP 290: Filter Incoming Serialization Data](https://openjdk.java.net/jeps/290)
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
