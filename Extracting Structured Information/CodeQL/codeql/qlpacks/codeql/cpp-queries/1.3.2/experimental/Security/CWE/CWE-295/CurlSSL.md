# Disabled certifcate verification
Disabling verification of the SSL certificate allows man-in-the-middle attacks. A SSL connection is vulnerable to man-in-the-middle attacks if the certification is not checked properly. If the peer or the host's certificate verification is not verified, the underlying SSL communication is insecure.


## Recommendation
It is recommended that all communications be done post verification of the host as well as the peer.


## Example
The following snippet disables certification verification by setting the value of ` CURLOPT_SSL_VERIFYHOST` and `CURLOPT_SSL_VERIFYHOST` to `0`:


```cpp
string host = "codeql.com"
void bad(void) {
	std::unique_ptr<CURL, void(*)(CURL*)> curl =
		std::unique_ptr<CURL, void(*)(CURL*)>(curl_easy_init(), curl_easy_cleanup);
	curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 0);
	curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 0); 
  	curl_easy_setopt(curl.get(), CURLOPT_URL, host.c_str());
  	curl_easy_perform(curl.get());
}
```
This is bad as the certificates are not verified any more. This can be easily fixed by setting the values of the options to `2`.


```cpp
string host = "codeql.com"
void good(void) {
	std::unique_ptr<CURL, void(*)(CURL*)> curl =
		std::unique_ptr<CURL, void(*)(CURL*)>(curl_easy_init(), curl_easy_cleanup);
	curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 2);
	curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2); 
  	curl_easy_setopt(curl.get(), CURLOPT_URL, host.c_str());
  	curl_easy_perform(curl.get());
}
```

## References
* Curl Documentation:[ CURLOPT_SSL_VERIFYHOST](https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html)
* Curl Documentation:[ CURLOPT_SSL_VERIFYPEER](https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html)
* Related CVE: [ CVE-2022-33684](https://github.com/advisories/GHSA-5r3h-c3r7-9w4h)
* Related security advisory: [ openframeworks/openframeworks ](https://huntr.com/bounties/42325662-6329-4e04-875a-49e2f5d69f78)
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
