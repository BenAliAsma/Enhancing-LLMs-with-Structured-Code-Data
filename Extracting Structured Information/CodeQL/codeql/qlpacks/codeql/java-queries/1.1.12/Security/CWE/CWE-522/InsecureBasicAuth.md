# Insecure basic authentication
Basic authentication only obfuscates usernames and passwords in Base64 encoding, which can be easily recognized and reversed, thus it must not be transmitted over the cleartext HTTP channel. Transmitting sensitive information without using HTTPS makes the data vulnerable to packet sniffing.


## Recommendation
Either use a more secure authentication mechanism like digest authentication or federated authentication, or use the HTTPS communication protocol.


## Example
The following example shows two ways of using basic authentication. In the 'BAD' case, the credentials are transmitted over HTTP. In the 'GOOD' case, the credentials are transmitted over HTTPS.


```java
public class InsecureBasicAuth {
  /**
   * Test basic authentication with Apache HTTP request.
   */
  public void testApacheHttpRequest(String username, String password) {

    // BAD: basic authentication over HTTP
    String url = "http://www.example.com/rest/getuser.do?uid=abcdx";

    // GOOD: basic authentication over HTTPS
    url = "https://www.example.com/rest/getuser.do?uid=abcdx";

    HttpPost post = new HttpPost(url);
    post.setHeader("Accept", "application/json");
    post.setHeader("Content-type", "application/json");

    String authString = username + ":" + password;
    byte[] authEncBytes = Base64.getEncoder().encode(authString.getBytes());
    String authStringEnc = new String(authEncBytes);

    post.addHeader("Authorization", "Basic " + authStringEnc);
  }

  /**
   * Test basic authentication with Java HTTP URL connection.
   */
  public void testHttpUrlConnection(String username, String password) {

    // BAD: basic authentication over HTTP
    String urlStr = "http://www.example.com/rest/getuser.do?uid=abcdx";

    // GOOD: basic authentication over HTTPS
    urlStr = "https://www.example.com/rest/getuser.do?uid=abcdx";

    String authString = username + ":" + password;
    String encoding = Base64.getEncoder().encodeToString(authString.getBytes("UTF-8"));
    URL url = new URL(urlStr);
    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
    conn.setRequestMethod("POST");
    conn.setDoOutput(true);
    conn.setRequestProperty("Authorization", "Basic " + encoding);
  }
}

```

## References
* SonarSource rule: [Basic authentication should not be used](https://rules.sonarsource.com/java/tag/owasp/RSPEC-2647).
* Acunetix: [WEB VULNERABILITIES INDEX - Basic authentication over HTTP](https://www.acunetix.com/vulnerabilities/web/basic-authentication-over-http/).
* Common Weakness Enumeration: [CWE-522](https://cwe.mitre.org/data/definitions/522.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
