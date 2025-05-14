# Timing attack against header value
A constant-time algorithm should be used for checking the value of sensitive headers. In other words, the comparison time should not depend on the content of the input. Otherwise timing information could be used to infer the header's expected, secret value.


## Recommendation
Use `MessageDigest.isEqual()` method to check the value of headers. If this method is used, then the calculation time depends only on the length of input byte arrays, and does not depend on the contents of the arrays.


## Example
The following example uses `String.equals()` method for validating a csrf token. This method implements a non-constant-time algorithm. The example also demonstrates validation using a safe constant-time algorithm.


```java
import javax.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.lang.String;


public class Test {
    private boolean UnsafeComparison(HttpServletRequest request) {
        String Key = "secret";
        return Key.equals(request.getHeader("X-Auth-Token"));        
    }

    private boolean safeComparison(HttpServletRequest request) {
          String token = request.getHeader("X-Auth-Token");
          String Key = "secret"; 
          return MessageDigest.isEqual(Key.getBytes(StandardCharsets.UTF_8), token.getBytes(StandardCharsets.UTF_8));
    }
    
}


```
