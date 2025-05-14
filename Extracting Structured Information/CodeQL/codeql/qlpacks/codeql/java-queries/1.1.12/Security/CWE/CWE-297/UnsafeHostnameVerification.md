# Unsafe hostname verification
If a `HostnameVerifier` always returns `true` it will not verify the hostname at all. This stops Transport Layer Security (TLS) providing any security and allows an attacker to perform a man-in-the-middle attack against the application.

An attack might look like this:

1. The program connects to `https://example.com`.
1. The attacker intercepts this connection and presents an apparently-valid certificate of their choosing.
1. The `TrustManager` of the program verifies that the certificate has been issued by a trusted certificate authority.
1. The Java HTTPS library checks whether the certificate has been issued for the host `example.com`. This check fails because the certificate has been issued for a domain controlled by the attacker, for example: `malicious.domain`.
1. The HTTPS library wants to reject the certificate because the hostname does not match. Before doing this it checks whether a `HostnameVerifier` exists.
1. Your `HostnameVerifier` is called which returns `true` for any certificate so also for this one.
1. The program proceeds with the connection since your `HostnameVerifier` accepted it.
1. The attacker can now read the data your program sends to `https://example.com` and/or alter its replies while the program thinks the connection is secure.

## Recommendation
Do not use an open `HostnameVerifier`. If you have a configuration problem with TLS/HTTPS, you should always solve the configuration problem instead of using an open verifier.


## Example
In the first (bad) example, the `HostnameVerifier` always returns `true`. This allows an attacker to perform a man-in-the-middle attack, because any certificate is accepted despite an incorrect hostname. In the second (good) example, the `HostnameVerifier` only returns `true` when the certificate has been correctly checked.


```java
public static void main(String[] args) {

	{
		HostnameVerifier verifier = new HostnameVerifier() {
			@Override
			public boolean verify(String hostname, SSLSession session) {
				return true; // BAD: accept even if the hostname doesn't match
			}
		};
		HttpsURLConnection.setDefaultHostnameVerifier(verifier);
	}

	{
		HostnameVerifier verifier = new HostnameVerifier() {
			@Override
			public boolean verify(String hostname, SSLSession session) {
				try { // GOOD: verify the certificate
					Certificate[] certs = session.getPeerCertificates();
					X509Certificate x509 = (X509Certificate) certs[0];
					check(new String[]{host}, x509);
					return true;
				} catch (SSLException e) {
					return false;
				}
			}
		};
		HttpsURLConnection.setDefaultHostnameVerifier(verifier);
	}

}
```

## References
* Android developers: [Security with HTTPS and SSL](https://developer.android.com/training/articles/security-ssl).
* Terse systems blog: [Fixing Hostname Verification](https://tersesystems.com/blog/2014/03/23/fixing-hostname-verification/).
* Common Weakness Enumeration: [CWE-297](https://cwe.mitre.org/data/definitions/297.html).
