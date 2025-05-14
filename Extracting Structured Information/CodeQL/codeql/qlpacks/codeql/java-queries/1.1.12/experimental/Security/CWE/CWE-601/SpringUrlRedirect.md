# Spring url redirection from remote source
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but which is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.


## Example
The following examples show the bad case and the good case respectively. The `bad` methods show an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks. In the `good1` method, it is shown how to solve this problem by verifying whether the user input is a known fixed string beginning.


```java
import javax.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.view.RedirectView;

@Controller
public class SpringUrlRedirect {

    private final static String VALID_REDIRECT = "http://127.0.0.1";

    @GetMapping("url1")
    public RedirectView bad1(String redirectUrl, HttpServletResponse response) throws Exception {
        RedirectView rv = new RedirectView();
        rv.setUrl(redirectUrl);
        return rv;
    }

    @GetMapping("url2")
    public String bad2(String redirectUrl) {
        String url = "redirect:" + redirectUrl;
        return url;
    }

    @GetMapping("url3")
    public RedirectView bad3(String redirectUrl) {
        RedirectView rv = new RedirectView(redirectUrl);
        return rv;
    }

    @GetMapping("url4")
    public ModelAndView bad4(String redirectUrl) {
        return new ModelAndView("redirect:" + redirectUrl);
    }

    @GetMapping("url5")
    public ResponseEntity<Void> bad5(String redirectUrl) {
        return ResponseEntity.status(HttpStatus.FOUND)
                .location(URI.create(redirectUrl))
                .build();
    }

    @GetMapping("url6")
    public ResponseEntity<Void> bad6(String redirectUrl) {
        HttpHeaders httpHeaders = new HttpHeaders();
        httpHeaders.setLocation(URI.create(redirectUrl));

        return new ResponseEntity<>(httpHeaders, HttpStatus.SEE_OTHER);
    }

    @GetMapping("url7")
    public ResponseEntity<Void> bad7(String redirectUrl) {
        HttpHeaders httpHeaders = new HttpHeaders();
        httpHeaders.add("Location", redirectUrl);

        return ResponseEntity.status(HttpStatus.SEE_OTHER).headers(httpHeaders).build();
    }

    @GetMapping("url8")
    public RedirectView good1(String redirectUrl) {
        RedirectView rv = new RedirectView();
        if (redirectUrl.startsWith(VALID_REDIRECT)){
            rv.setUrl(redirectUrl);
        }else {
            rv.setUrl(VALID_REDIRECT);
        }
        return rv;
    }
}

```

## References
* A Guide To Spring Redirects: [Spring Redirects](https://www.baeldung.com/spring-redirect-and-forward).
* Url redirection - attack and defense: [Url Redirection](https://www.virtuesecurity.com/kb/url-redirection-attack-and-defense/).
* How to redirect to an external URL from Spring Boot REST Controller (Post/Redirect/Get pattern)?: [ResponseEntity Redirection](https://fullstackdeveloper.guru/2021/03/12/how-to-redirect-to-an-external-url-from-spring-boot-rest-controller/).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).
