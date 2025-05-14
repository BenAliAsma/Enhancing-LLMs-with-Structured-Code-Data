# Use of insufficient randomness as the key of a cryptographic algorithm
Using a cryptographically weak pseudo-random number generator to generate a security-sensitive value, such as a password, makes it easier for an attacker to predict the value.

Pseudo-random number generators generate a sequence of numbers that only approximates the properties of random numbers. The sequence is not truly random because it is completely determined by a relatively small set of initial values, the seed. If the random number generator is cryptographically weak, then this sequence may be easily predictable through outside observations.


## Recommendation
Use a cryptographically secure pseudo-random number generator if the output is to be used in a security sensitive context. As a rule of thumb, a value should be considered "security sensitive" if predicting it would allow the attacker to perform an action that they would otherwise be unable to perform. For example, if an attacker could predict the random password generated for a new user, they would be able to log in as that new user.

For Go, `crypto/rand` provides a cryptographically secure pseudo-random number generator. `math/rand` is not cryptographically secure, and should be avoided in security contexts. For contexts which are not security sensitive, `math/rand` may be preferable as it has a more convenient interface, and is likely to be faster.


## Example
The example below uses the `math/rand` package instead of `crypto/rand` to generate a password:


```go
package main

import (
	"math/rand"
)

var charset = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

func generatePassword() string {
	s := make([]rune, 20)
	for i := range s {
		s[i] = charset[rand.Intn(len(charset))]
	}
	return string(s)
}

```
Instead, use `crypto/rand`:


```go
package main

import (
	"crypto/rand"
	"math/big"
)

func generatePasswordGood() string {
	s := make([]rune, 20)
	for i := range s {
		idx, err := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
		if err != nil {
			// handle err
		}
		s[i] = charset[idx.Int64()]
	}
	return string(s)
}

```

## References
* Wikipedia. [Pseudo-random number generator](http://en.wikipedia.org/wiki/Pseudorandom_number_generator).
* OWASP: [Insecure Randomness](https://owasp.org/www-community/vulnerabilities/Insecure_Randomness).
* OWASP: [Secure Random Number Generation](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html#secure-random-number-generation).
* Common Weakness Enumeration: [CWE-338](https://cwe.mitre.org/data/definitions/338.html).
