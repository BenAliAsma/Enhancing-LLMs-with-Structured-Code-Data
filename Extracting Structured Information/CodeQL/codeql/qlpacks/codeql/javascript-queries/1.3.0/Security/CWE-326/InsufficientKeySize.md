# Use of a weak cryptographic key
Modern encryption relies on it being computationally infeasible to break the cipher and decode a message without the key. As computational power increases, the ability to break ciphers grows and keys need to become larger.


## Recommendation
An encryption key should be at least 2048-bit long when using RSA encryption, and 128-bit long when using symmetric encryption.


## References
* Wikipedia: [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)).
* Wikipedia: [AES](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard).
* NodeJS: [Crypto](https://nodejs.org/api/crypto.html).
* NIST: [ Recommendation for Transitioning the Use of Cryptographic Algorithms and Key Lengths](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar1.pdf).
* Wikipedia: [Key size](https://en.wikipedia.org/wiki/Key_size)
* Common Weakness Enumeration: [CWE-326](https://cwe.mitre.org/data/definitions/326.html).
