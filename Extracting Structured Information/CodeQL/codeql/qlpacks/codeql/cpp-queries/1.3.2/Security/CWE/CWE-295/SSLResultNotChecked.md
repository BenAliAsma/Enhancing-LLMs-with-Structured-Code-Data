# Certificate not checked
After fetching an SSL certificate, always check the result of certificate verification.


## Recommendation
Always check the result of SSL certificate verification. A certificate that has been revoked may indicate that data is coming from an attacker, whereas a certificate that has expired or was self-signed may indicate an increased likelihood that the data is malicious.


## Example
In this example, the `SSL_get_peer_certificate` function is used to get the certificate of a peer. However it is unsafe to use that information without checking if the certificate is valid.


```cpp
// ...

X509 *cert = SSL_get_peer_certificate(ssl); // BAD (SSL_get_verify_result is never called)

// ...
```
In the corrected example, we use `SSL_get_verify_result` to check that certificate verification was successful.


```cpp
// ...

X509 *cert = SSL_get_peer_certificate(ssl); // GOOD
if (cert)
{
	result = SSL_get_verify_result(ssl);
	if (result == X509_V_OK)
	{
		// ...
```

## References
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
