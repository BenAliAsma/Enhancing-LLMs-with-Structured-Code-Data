# Use of a weak cryptographic key
Incorrect uses of encryption algorithms may result in sensitive data exposure, key leakage, broken authentication, insecure session, and spoofing attacks.


## Recommendation
Ensure that you use a strong key with a recommended bit size. For RSA encryption the minimum size is 2048 bits.


## Example
The following code uses RSA encryption with insufficient key size.


```go
package main

import (
	"crypto/rand"
	"crypto/rsa"
	"fmt"
)

func main() {
	//Generate Private Key
	pvk, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(pvk)
}

```
In the example below, the key size is set to 2048 bits.


```go
package main

import (
	"crypto/rand"
	"crypto/rsa"
	"fmt"
)

func main() {
	//Generate Private Key
	pvk, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(pvk)
}

```

## References
* OWASP: [Cryptographic Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html).
* Wikipedia: [Cryptographically Strong Algorithms](https://en.wikipedia.org/wiki/Strong_cryptography#Cryptographically_strong_algorithms).
* Wikipedia: [Strong Cryptography Examples](https://en.wikipedia.org/wiki/Strong_cryptography#Examples).
* NIST, FIPS 140 Annex a: [ Approved Security Functions](http://csrc.nist.gov/publications/fips/fips140-2/fips1402annexa.pdf).
* NIST, SP 800-131A: [ Transitions: Recommendation for Transitioning the Use of Cryptographic Algorithms and Key Lengths](http://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar1.pdf).
* Common Weakness Enumeration: [CWE-326](https://cwe.mitre.org/data/definitions/326.html).
