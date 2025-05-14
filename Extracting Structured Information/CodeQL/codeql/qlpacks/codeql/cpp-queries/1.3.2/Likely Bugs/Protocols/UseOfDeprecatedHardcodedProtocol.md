# boost::asio use of deprecated hardcoded protocol
Using boost::asio library but specifying a deprecated hardcoded protocol.


## Recommendation
Only use modern protocols such as TLS 1.2 or TLS 1.3.


## Example
In the following example, the `sslv2` protocol is specified. This protocol is out of date and its use is not recommended.


```cpp

void useProtocol_bad()
{
	boost::asio::ssl::context ctx_sslv2(boost::asio::ssl::context::sslv2); // BAD: outdated protocol

	// ...
}

```
In the corrected example, the `tlsv13` protocol is used instead.


```cpp

void useProtocol_good()
{
	boost::asio::ssl::context cxt_tlsv13(boost::asio::ssl::context::tlsv13);

	// ...
}

```

## References
* [Boost.Asio documentation](https://www.boost.org/doc/libs/1_71_0/doc/html/boost_asio.html).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
