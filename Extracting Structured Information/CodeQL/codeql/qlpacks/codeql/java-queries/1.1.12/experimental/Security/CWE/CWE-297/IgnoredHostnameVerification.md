# Ignored result of hostname verification
The method `HostnameVerifier.verify()` checks that the hostname from the server's certificate matches the server hostname after an HTTPS connection is established. The method returns `true` if the hostname is acceptable and `false` otherwise. The contract of the method does not require it to throw an exception if the verification failed. Therefore, a caller has to check the result and drop the connection if the hostname verification failed. Otherwise, an attacker may be able to implement a man-in-the-middle attack and impersonate the server.


## Recommendation
Always check the result of `HostnameVerifier.verify()` and drop the connection if the method returns false.


## Example
In the following example, the method `HostnameVerifier.verify()` is called but its result is ignored. As a result, no hostname verification actually happens.


```java
public SSLSocket connect(String host, int port, HostnameVerifier verifier) {
    SSLSocket socket = (SSLSocket) SSLSocketFactory.getDefault().createSocket(host, port);
    socket.startHandshake();
    verifier.verify(host, socket.getSession());
    return socket;
}
```
In the next example, the result of the `HostnameVerifier.verify()` method is checked and an exception is thrown if the verification failed.


```java
public SSLSocket connect(String host, int port, HostnameVerifier verifier) {
    SSLSocket socket = (SSLSocket) SSLSocketFactory.getDefault().createSocket(host, port);
    socket.startHandshake();
    boolean successful = verifier.verify(host, socket.getSession());
    if (!successful) {
        socket.close();
        throw new SSLException("Oops! Hostname verification failed!");
    }
    return socket;
}
```

## References
* Java API Specification: [HostnameVerifier.verify() method](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/HostnameVerifier.html#verify(java.lang.String,javax.net.ssl.SSLSession)).
* Common Weakness Enumeration: [CWE-297](https://cwe.mitre.org/data/definitions/297.html).
