# Spring View Manipulation
The Spring Expression Language (SpEL) is a powerful expression language provided by Spring Framework. The language offers many features including invocation of methods available in the JVM.

An unrestricted view name manipulation vulnerability in Spring Framework could lead to attacker-controlled arbitrary SpEL expressions being evaluated using attacker-controlled data, which may in turn allow an attacker to run arbitrary code.

Note: two related variants of this problem are detected by different queries, \`java/spring-view-manipulation\` and \`java/spring-view-manipulation-implicit\`. The first detects taint flow problems where the return types is always `String`. While the latter, \`java/spring-view-manipulation-implicit\` detects cases where the request mapping method has a non-string return type such as `void`.


## Recommendation
In general, using user input to determine Spring view name should be avoided. If user input must be included in the expression, the controller can be annotated by a `@ResponseBody` annotation. In this case, Spring Framework does not interpret it as a view name, but just returns this string in HTTP Response. The same applies to using a `@RestController` annotation on a class, as internally it inherits `@ResponseBody`.


## Example
In the following example, the `Fragment` method uses an externally controlled variable `section` to generate the view name. Hence, it is vulnerable to Spring View Manipulation attacks.


```java
@Controller
public class SptingViewManipulationController {

    Logger log = LoggerFactory.getLogger(HelloController.class);

    @GetMapping("/safe/fragment")
    public String Fragment(@RequestParam String section) {
        // bad as template path is attacker controlled
        return "welcome :: " + section;
    }

    @GetMapping("/doc/{document}")
    public void getDocument(@PathVariable String document) {
        // returns void, so view name is taken from URI
        log.info("Retrieving " + document);
    }
}

```
This can be easily prevented by using the `ResponseBody` annotation which marks the response is already processed preventing exploitation of Spring View Manipulation vulnerabilities. Alternatively, this can also be fixed by adding a `HttpServletResponse` parameter to the method definition as shown in the example below.


```java
@Controller
public class SptingViewManipulationController {

    Logger log = LoggerFactory.getLogger(HelloController.class);

    @GetMapping("/safe/fragment")
    @ResponseBody
    public String Fragment(@RequestParam String section) {
        // good, as `@ResponseBody` annotation tells Spring
        // to process the return values as body, instead of view name
        return "welcome :: " + section;
    }

    @GetMapping("/safe/doc/{document}")
    public void getDocument(@PathVariable String document, HttpServletResponse response) {
        // good as `HttpServletResponse param tells Spring that the response is already
        // processed.
        log.info("Retrieving " + document); // FP
    }
}

```

## References
* Veracode Research : [Spring View Manipulation ](https://github.com/veracode-research/spring-view-manipulation/)
* Spring Framework Reference Documentation: [Spring Expression Language (SpEL)](https://docs.spring.io/spring/docs/4.2.x/spring-framework-reference/html/expressions.html)
* OWASP: [Expression Language Injection](https://owasp.org/www-community/vulnerabilities/Expression_Language_Injection)
* Common Weakness Enumeration: [CWE-94](https://cwe.mitre.org/data/definitions/94.html).
