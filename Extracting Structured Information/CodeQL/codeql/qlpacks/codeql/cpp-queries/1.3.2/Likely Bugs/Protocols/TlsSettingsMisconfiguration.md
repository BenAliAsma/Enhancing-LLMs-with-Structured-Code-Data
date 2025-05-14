# boost::asio TLS settings misconfiguration
Using the TLS or SSLv23 protocol from the boost::asio library, but not disabling deprecated protocols may expose the software to known vulnerabilities or permit weak encryption algorithms to be used. Disabling the minimum-recommended protocols is also flagged.


## Recommendation
When using the TLS or SSLv23 protocol, set the `no_tlsv1` and `no_tlsv1_1` options, but do not set `no_tlsv1_2`. When using the SSLv23 protocol, also set the `no_sslv3` option.


## Example
In the following example, the `no_tlsv1_1` option has not been set. Use of TLS 1.1 is not recommended.


```cpp

void useTLS_bad()
{
	boost::asio::ssl::context ctx(boost::asio::ssl::context::tls);
	ctx.set_options(boost::asio::ssl::context::no_tlsv1); // BAD: missing no_tlsv1_1

	// ...
}

```
In the corrected example, the `no_tlsv1` and `no_tlsv1_1` options have both been set, ensuring the use of TLS 1.2 or later.


```cpp

void useTLS_good()
{
	boost::asio::ssl::context ctx(boost::asio::ssl::context::tls);
	ctx.set_options(boost::asio::ssl::context::no_tlsv1 | boost::asio::ssl::context::no_tlsv1_1); // GOOD

	// ...
}

```

## References
* [Boost.Asio documentation](https://www.boost.org/doc/libs/1_71_0/doc/html/boost_asio.html).
* Common Weakness Enumeration: [CWE-326](https://cwe.mitre.org/data/definitions/326.html).
