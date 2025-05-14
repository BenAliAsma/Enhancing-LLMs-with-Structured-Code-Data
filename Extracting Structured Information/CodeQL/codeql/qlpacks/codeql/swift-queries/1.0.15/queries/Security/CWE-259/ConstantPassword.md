# Constant password
Deriving password-based encryption keys using hardcoded passwords is insecure, because the generated key may be easily discovered. Data hashed using constant salts is vulnerable to dictionary attacks, enabling attackers to recover the original input.

In particular, constant passwords would enable easier recovery of the key, even in the presence of a salt. If that salt is random enough, then key recovery is not as easy as just looking up a hardcoded credential in the source code.


## Recommendation
Use randomly generated passwords to securely derive a password-based encryption key.


## Example
The following example shows a few cases of hashing input data. In the 'BAD' cases, the password is constant, making the derived key vulnerable to dictionary attakcs. In the 'GOOD' cases, the password is randomly generated, which protects the hashed data against recovery.


```swift

func encrypt(padding : Padding) {
	// ...

	// BAD: Using constant passwords for hashing
	let password: Array<UInt8> = [0x2a, 0x3a, 0x80, 0x05]
	let randomArray = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	_ = try HKDF(password: password, salt: randomArray, info: randomArray, keyLength: 0, variant: Variant.sha2)
	_ = try PKCS5.PBKDF1(password: password, salt: randomArray, iterations: 120120, keyLength: 0)
	_ = try PKCS5.PBKDF2(password: password, salt: randomArray, iterations: 120120, keyLength: 0)
	_ = try Scrypt(password: password, salt: randomArray, dkLen: 64, N: 16384, r: 8, p: 1)

	// GOOD: Using randomly generated passwords for hashing
	let password = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	let randomArray = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	_ = try HKDF(password: password, salt: randomArray, info: randomArray, keyLength: 0, variant: Variant.sha2)
	_ = try PKCS5.PBKDF1(password: password, salt: randomArray, iterations: 120120, keyLength: 0)
	_ = try PKCS5.PBKDF2(password: password, salt: randomArray, iterations: 120120, keyLength: 0)
	_ = try Scrypt(password: password, salt: randomArray, dkLen: 64, N: 16384, r: 8, p: 1)

	// ...
}

```

## References
* Okta blog: [What are Salted Passwords and Password Hashing?](https://www.okta.com/blog/2019/03/what-are-salted-passwords-and-password-hashing/)
* RFC 2898: [Password-Based Cryptography Specification](https://www.rfc-editor.org/rfc/rfc2898).
* Common Weakness Enumeration: [CWE-259](https://cwe.mitre.org/data/definitions/259.html).
