# Weak encryption: Insufficient key size
This rule finds uses of encryption algorithms with too small a key size. Encryption algorithms are vulnerable to brute force attack when too small a key size is used.


## Recommendation
The key should be at least 2048-bit long when using RSA encryption, and 128-bit long when using symmetric encryption.


## References
* Wikipedia. [Key size](http://en.wikipedia.org/wiki/Key_size).
* Common Weakness Enumeration: [CWE-326](https://cwe.mitre.org/data/definitions/326.html).
