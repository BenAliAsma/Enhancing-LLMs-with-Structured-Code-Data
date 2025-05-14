# Unsafe certificate trust
Java offers two mechanisms for SSL authentication - trust manager and hostname verifier (the later is checked by the `java/insecure-hostname-verifier` query). The trust manager validates the peer's certificate chain while hostname verification establishes that the hostname in the URL matches the hostname in the server's identification.

When `SSLSocket` or `SSLEngine` are created without a secure `setEndpointIdentificationAlgorithm`, hostname verification is disabled by default.

This query checks whether `setEndpointIdentificationAlgorithm` is missing, thereby making the application vulnerable to man-in-the-middle attacks. The query also covers insecure configurations of `com.rabbitmq.client.ConnectionFactory`.


## Recommendation
Validate SSL certificates in SSL authentication.


## Example
The following two examples show two ways of configuring SSLSocket/SSLEngine. In the 'BAD' case, `setEndpointIdentificationAlgorithm` is not called, thus no hostname verification takes place. In the 'GOOD' case, `setEndpointIdentificationAlgorithm` is called.


```java
public static void main(String[] args) {

	{
		SSLContext sslContext = SSLContext.getInstance("TLS");
		SSLEngine sslEngine = sslContext.createSSLEngine();
		SSLParameters sslParameters = sslEngine.getSSLParameters();
		sslParameters.setEndpointIdentificationAlgorithm("HTTPS"); //GOOD: Set a valid endpointIdentificationAlgorithm for SSL engine to trigger hostname verification
		sslEngine.setSSLParameters(sslParameters);
	}

	{
		SSLContext sslContext = SSLContext.getInstance("TLS");
		SSLEngine sslEngine = sslContext.createSSLEngine();  //BAD: No endpointIdentificationAlgorithm set
	}

	{
		SSLContext sslContext = SSLContext.getInstance("TLS");
		final SSLSocketFactory socketFactory = sslContext.getSocketFactory();
		SSLSocket socket = (SSLSocket) socketFactory.createSocket("www.example.com", 443); 
		SSLParameters sslParameters = sslEngine.getSSLParameters();
		sslParameters.setEndpointIdentificationAlgorithm("HTTPS"); //GOOD: Set a valid endpointIdentificationAlgorithm for SSL socket to trigger hostname verification
		socket.setSSLParameters(sslParameters);
	}

	{
		com.rabbitmq.client.ConnectionFactory connectionFactory = new com.rabbitmq.client.ConnectionFactory();
		connectionFactory.useSslProtocol();
		connectionFactory.enableHostnameVerification();  //GOOD: Enable hostname verification for rabbitmq ConnectionFactory
	}

	{
		com.rabbitmq.client.ConnectionFactory connectionFactory = new com.rabbitmq.client.ConnectionFactory();
		connectionFactory.useSslProtocol(); //BAD: Hostname verification for rabbitmq ConnectionFactory is not enabled
	}
}
```

## References
* [Testing Endpoint Identify Verification (MSTG-NETWORK-3)](https://github.com/OWASP/owasp-mstg/blob/master/Document/0x05g-Testing-Network-Communication.md).
* [SSLParameters.setEndpointIdentificationAlgorithm documentation](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/SSLParameters.html#setEndpointIdentificationAlgorithm(java.lang.String)).
* RabbitMQ: [ConnectionFactory.enableHostnameVerification documentation](https://rabbitmq.github.io/rabbitmq-java-client/api/current/com/rabbitmq/client/ConnectionFactory.html#enableHostnameVerification()).
* RabbitMQ: [Using TLS in the Java Client](https://www.rabbitmq.com/ssl.html#java-client).
* [CVE-2018-17187: Apache Qpid Proton-J transport issue with hostname verification](https://github.com/advisories/GHSA-xvch-r4wf-h8w9).
* [CVE-2018-8034: Apache Tomcat - host name verification when using TLS with the WebSocket client](https://github.com/advisories/GHSA-46j3-r4pj-4835).
* [CVE-2018-11087: Pivotal Spring AMQP vulnerability due to lack of hostname validation](https://github.com/advisories/GHSA-w4g2-9hj6-5472).
* [CVE-2018-11775: TLS hostname verification issue when using the Apache ActiveMQ Client](https://github.com/advisories/GHSA-m9w8-v359-9ffr).
* Common Weakness Enumeration: [CWE-273](https://cwe.mitre.org/data/definitions/273.html).
