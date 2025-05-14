# Failure to use HTTPS URLs
Constructing URLs with the HTTP protocol can lead to unsecured connections.

Furthermore, constructing URLs with the HTTP protocol can create problems if other parts of the code expect HTTPS URLs. A typical pattern is to cast the `URLConnection` that is produced by `url.getConnection()` to an `HttpsURLConnection`. This is impossible if the URL that has been constructed uses HTTP rather than HTTPS, and results in a run-time `ClassCastException`.


## Recommendation
When you construct a URL using `java.net.URL`, ensure that you use an HTTPS URL rather than an HTTP URL. Then, any connections that are made using that URL are secure SSL connections.


## Example
The following example shows two ways of opening a connection using a URL. When the connection is opened using an HTTP URL rather than an HTTPS URL, the connection is unsecured, and in this case a `ClassCastException` is caused. When the connection is opened using an HTTPS URL, the connection is a secure SSL connection.


```java
public static void main(String[] args) {
	{
		try {
			String protocol = "http://";
			URL u = new URL(protocol + "www.secret.example.org/");
			// BAD: This causes a 'ClassCastException' at runtime, because the
			// HTTP URL cannot be used to make an 'HttpsURLConnection', 
			// which enforces SSL.
			HttpsURLConnection hu = (HttpsURLConnection) u.openConnection();
			hu.setRequestMethod("PUT");
			hu.connect();
			OutputStream os = hu.getOutputStream();
			hu.disconnect();
		}
		catch (IOException e) {
			// fail
		}
	}
	
	{
		try {
			String protocol = "https://";
			URL u = new URL(protocol + "www.secret.example.org/");
			// GOOD: Opening a connection to a URL using HTTPS enforces SSL.
			HttpsURLConnection hu = (HttpsURLConnection) u.openConnection();
			hu.setRequestMethod("PUT");
			hu.connect();
			OutputStream os = hu.getOutputStream();
			hu.disconnect();
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
* OWASP: [Transport Layer Protection Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
* Common Weakness Enumeration: [CWE-345](https://cwe.mitre.org/data/definitions/345.html).
