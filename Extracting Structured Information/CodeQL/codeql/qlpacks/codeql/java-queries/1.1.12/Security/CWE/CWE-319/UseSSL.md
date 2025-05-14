# Failure to use SSL
Using a stream that is derived from a non-SSL connection or socket can result in an unsecured connection that is vulnerable to interception.


## Recommendation
Use `javax.net.ssl.HttpsURLConnection` and `javax.net.ssl.SSLSocket` instead of the corresponding unsecured versions in `java.net`. If necessary, downcast from an `HttpURLConnection` to an `HttpsURLConnection` to enforce the use of SSL. In addition, when you construct a `java.net.URL`, ensure that you use the HTTPS protocol, to avoid exceptions when trying to make HTTPS connections to the URL.


## Example
The following example shows two ways of opening an output stream. When the stream is opened using `httpcon`, which is an `HttpURLConnection`, the connection does not use SSL, and therefore is vulnerable to attack. When the stream is opened using `httpscon`, the connection is a secured SSL connection.


```java
public static void main(String[] args) {
	{
		try {
			URL u = new URL("http://www.secret.example.org/");
			HttpURLConnection httpcon = (HttpURLConnection) u.openConnection();
			httpcon.setRequestMethod("PUT");
			httpcon.connect();
			// BAD: output stream from non-HTTPS connection
			OutputStream os = httpcon.getOutputStream();
			httpcon.disconnect();
		}
		catch (IOException e) {
			// fail
		}
	}
	
	{
		try {
			URL u = new URL("https://www.secret.example.org/");
			HttpsURLConnection httpscon = (HttpsURLConnection) u.openConnection();
			httpscon.setRequestMethod("PUT");
			httpscon.connect();
			// GOOD: output stream from HTTPS connection
			OutputStream os = httpscon.getOutputStream();
			httpscon.disconnect();
		}
		catch (IOException e) {
			// fail
		}
	}
}
```

## References
* SEI CERT Oracle Coding Standard for Java: [SER03-J. Do not serialize unencrypted, sensitive data](https://wiki.sei.cmu.edu/confluence/display/java/SER03-J.+Do+not+serialize+unencrypted+sensitive+data).
* Java API Specification: [ Class HttpsURLConnection](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/HttpsURLConnection.html).
* Java API Specification: [ Class SSLSocket](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/javax/net/ssl/SSLSocket.html).
* OWASP: [Transport Layer Protection Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
