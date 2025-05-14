# Certificate result conflation
When checking the result of SSL certificate verification, accepting any error code may allow an attacker to impersonate someone who is trusted.


## Recommendation
When checking an SSL certificate with `SSL_get_verify_result`, only `X509_V_OK` is a success code. If there is any other result the certificate should not be accepted.


## Example
In this example the error code `X509_V_ERR_CERT_HAS_EXPIRED` is treated the same as an OK result. An expired certificate should not be accepted as it is more likely to be compromised than a valid certificate.


```cpp
// ...

if (cert = SSL_get_peer_certificate(ssl))
{
	result = SSL_get_verify_result(ssl);

	if ((result == X509_V_OK) || (result == X509_V_ERR_CERT_HAS_EXPIRED)) // BAD (conflates OK and a non-OK codes)
	{
		do_ok();
	} else {
		do_error();
	}
}

```
In the corrected example, only a result of `X509_V_OK` is accepted.


```cpp
// ...

if (cert = SSL_get_peer_certificate(ssl))
{
	result = SSL_get_verify_result(ssl);

	if (result == X509_V_OK) // GOOD
	{
		do_ok();
	} else {
		do_error();
	}
}

```

## References
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
