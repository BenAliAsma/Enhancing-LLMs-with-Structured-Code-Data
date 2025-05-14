# Static initialization vector for encryption
When a cipher is used in certain modes (such as CBC or GCM), it requires an initialization vector (IV). Under the same secret key, IVs should be unique and ideally unpredictable. If the same IV is used with the same secret key, then the same plaintext results in the same ciphertext. This behavior may enable an attacker to learn if the same data pieces are transferred or stored, or help the attacker run a dictionary attack.

In particular, if the IV is hardcoded or constant, an attacker may just look up potential keys in a dictionary, then concatenate those with the hardcoded or constant IV rather than trying to discover the entire encryption key.


## Recommendation
Use a randomly generated IV.


## Example
The following example shows a few cases of instantiating a cipher with various encryption keys. In the 'BAD' cases, the IV is hardcoded or constant, making the encrypted data vulnerable to recovery. In the 'GOOD' cases, the IV is randomly generated and not hardcoded, which protects the encrypted data against recovery.


```swift

func encrypt(padding : Padding) {
	// ...

	// BAD: Using static IVs for encryption
	let iv: Array<UInt8> = [0x2a, 0x3a, 0x80, 0x05]
	let ivString = "this is a constant string"
	let key = getRandomKey()
	_ = try AES(key: key, iv: ivString)
	_ = try CBC(iv: iv)

	// GOOD: Using randomly generated IVs for encryption
	let iv = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	let ivString = String(cString: iv)
	let key = getRandomKey())
	_ = try AES(key: key, iv: ivString)
	_ = try CBC(iv: iv)

	// ...
}

```

## References
* Wikipedia: [Initialization vector](https://en.wikipedia.org/wiki/Initialization_vector).
* National Institute of Standards and Technology: [Recommendation for Block Cipher Modes of Operation](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf).
* National Institute of Standards and Technology: [FIPS 140-2: Security Requirements for Cryptographic Modules](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.140-2.pdf).
* Common Weakness Enumeration: [CWE-329](https://cwe.mitre.org/data/definitions/329.html).
* Common Weakness Enumeration: [CWE-1204](https://cwe.mitre.org/data/definitions/1204.html).
