# Unsafe TLS version
Transport Layer Security (TLS) provides a number of security features such as confidentiality, integrity, replay prevention and authentication. There are several versions of TLS protocols. The latest is TLS 1.3. Unfortunately, older versions were found to be vulnerable to a number of attacks.


## Recommendation
An application should use TLS 1.3. Currently, TLS 1.2 is also considered acceptable.


## Example
The following example shows how a socket with an unsafe TLS version may be created:


```java
public SSLSocket connect(String host, int port)
        throws NoSuchAlgorithmException, IOException {
    
    SSLContext context = SSLContext.getInstance("SSLv3");
    return (SSLSocket) context.getSocketFactory().createSocket(host, port);
}
```
The next example creates a socket with the latest TLS version:


```java
public SSLSocket connect(String host, int port)
        throws NoSuchAlgorithmException, IOException {
    
    SSLContext context = SSLContext.getInstance("TLSv1.3");
    return (SSLSocket) context.getSocketFactory().createSocket(host, port);
}
```

## References
* Wikipedia: [Transport Layer Security](https://en.wikipedia.org/wiki/Transport_Layer_Security)
* OWASP: [Transport Layer Protection Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
* Java SE Documentation: [Java Secure Socket Extension (JSSE) Reference Guide](https://docs.oracle.com/javase/8/docs/technotes/guides/security/jsse/JSSERefGuide.html)
* Java API Specification: [SSLContext](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/SSLContext.html)
* Java API Specification: [SSLParameters](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/SSLParameters.html)
* Java API Specification: [SSLSocket](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/SSLSocket.html)
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
