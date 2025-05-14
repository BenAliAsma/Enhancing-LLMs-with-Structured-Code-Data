# Use of a weak cryptographic algorithm
Using weak cryptographic algorithms can leave data vulnerable to being decrypted or forged by an attacker.

Many cryptographic algorithms provided by cryptography libraries are known to be weak. Using such an algorithm means that encrypted or hashed data is less secure than it appears to be.


## Recommendation
Ensure that you use a strong, modern cryptographic algorithm. Use at least AES-128 or RSA-2048 for encryption, and SHA-2 or SHA-3 for secure hashing.


## Example
The following code uses the different packages to encrypt/hash some secret data. The first few examples uses DES, MD5, RC4, and SHA1, which are older algorithms that are now considered weak. The following examples use AES and SHA256, which are stronger, more modern algorithms.


```go
package main

import (
	"crypto/aes"
	"crypto/des"
	"crypto/md5"
	"crypto/rc4"
	"crypto/sha1"
	"crypto/sha256"
)

func main() {
	public := []byte("hello")

	password := []byte("123456")
	buf := password // testing dataflow by passing into different variable

	// BAD, des is a weak crypto algorithm and password is sensitive data
	des.NewTripleDESCipher(buf)

	// BAD, md5 is a weak crypto algorithm and password is sensitive data
	md5.Sum(buf)

	// BAD, rc4 is a weak crypto algorithm and password is sensitive data
	rc4.NewCipher(buf)

	// BAD, sha1 is a weak crypto algorithm and password is sensitive data
	sha1.Sum(buf)

	// GOOD, password is sensitive data but aes is a strong crypto algorithm
	aes.NewCipher(buf)

	// GOOD, password is sensitive data but sha256 is a strong crypto algorithm
	sha256.Sum256(buf)

	// GOOD, des is a weak crypto algorithm but public is not sensitive data
	des.NewTripleDESCipher(public)

	// GOOD, md5 is a weak crypto algorithm but public is not sensitive data
	md5.Sum(public)

	// GOOD, rc4 is a weak crypto algorithm but public is not sensitive data
	rc4.NewCipher(public)

	// GOOD, sha1 is a weak crypto algorithm but public is not sensitive data
	sha1.Sum(public)

	// GOOD, aes is a strong crypto algorithm and public is not sensitive data
	aes.NewCipher(public)

	// GOOD, sha256 is a strong crypto algorithm and public is not sensitive data
	sha256.Sum256(public)
}

```

## References
* OWASP: [Cryptographic Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html).
* Wikipedia: [Cryptographically Strong Algorithms](https://en.wikipedia.org/wiki/Strong_cryptography#Cryptographically_strong_algorithms).
* Wikipedia: [Strong Cryptography Examples](https://en.wikipedia.org/wiki/Strong_cryptography#Examples).
* NIST, FIPS 140 Annex a: [ Approved Security Functions](http://csrc.nist.gov/publications/fips/fips140-2/fips1402annexa.pdf).
* NIST, SP 800-131A: [ Transitions: Recommendation for Transitioning the Use of Cryptographic Algorithms and Key Lengths](http://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar1.pdf).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
* Common Weakness Enumeration: [CWE-328](https://cwe.mitre.org/data/definitions/328.html).
