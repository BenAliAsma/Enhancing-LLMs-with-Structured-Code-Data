# Encryption using ECB
ECB should not be used as a mode for encryption as it has dangerous weaknesses. Data is encrypted the same way every time, which means that the same plaintext input will always produce the same ciphertext. This behavior makes messages encrypted with ECB more vulnerable to replay attacks.


## Recommendation
Use a different cipher mode such as CBC.


## Example
The following example shows six cases of instantiating a cipher with various encryption keys and block modes. In the 'BAD' cases, the mode of encryption is ECB, making the encrypted data vulnerable to replay attacks. In the 'GOOD' cases, the encryption mode is CBC, which protects the encrypted data against replay attacks.


```swift

func encrypt(key : Key, padding : Padding) {
	// ...

	// BAD: ECB is used for block mode
	let blockMode = ECB()
	_ = try AES(key: key, blockMode: blockMode, padding: padding)
	_ = try AES(key: key, blockMode: blockMode)
	_ = try Blowfish(key: key, blockMode: blockMode, padding: padding)

	// GOOD: ECB is not used for block mode
	let aesBlockMode = CBC(iv: AES.randomIV(AES.blockSize))
	let blowfishBlockMode = CBC(iv: Blowfish.randomIV(Blowfish.blockSize))
	_ = try AES(key: key, blockMode: aesBlockMode, padding: padding)
	_ = try AES(key: key, blockMode: aesBlockMode)
	_ = try Blowfish(key: key, blockMode: blowfishBlockMode, padding: padding)

	// ...
}

```

## References
* Wikipedia, block cipher modes of operation, [Electronic codebook (ECB)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Electronic_codebook_.28ECB.29).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
