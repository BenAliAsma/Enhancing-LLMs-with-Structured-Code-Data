# Use of constant salts
Constant salts should not be used for password hashing. Data hashed using constant salts are vulnerable to dictionary attacks, enabling attackers to recover the original input.


## Recommendation
Use randomly generated salts to securely hash input data.


## Example
The following example shows a few cases of hashing input data. In the 'BAD' cases, the salt is constant, making the generated hashes vulnerable to dictionary attacks. In the 'GOOD' cases, the salt is randomly generated, which protects the hashed data against recovery.


```swift

func encrypt(padding : Padding) {
	// ...

	// BAD: Using constant salts for hashing
	let badSalt: Array<UInt8> = [0x2a, 0x3a, 0x80, 0x05]
	let randomArray = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	_ = try HKDF(password: randomArray, salt: badSalt, info: randomArray, keyLength: 0, variant: Variant.sha2)
	_ = try PKCS5.PBKDF1(password: randomArray, salt: badSalt, iterations: 120120, keyLength: 0)
	_ = try PKCS5.PBKDF2(password: randomArray, salt: badSalt, iterations: 120120, keyLength: 0)
	_ = try Scrypt(password: randomArray, salt: badSalt, dkLen: 64, N: 16384, r: 8, p: 1)

	// GOOD: Using randomly generated salts for hashing
	let goodSalt = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	let randomArray = (0..<10).map({ _ in UInt8.random(in: 0...UInt8.max) })
	_ = try HKDF(password: randomArray, salt: goodSalt, info: randomArray, keyLength: 0, variant: Variant.sha2)
	_ = try PKCS5.PBKDF1(password: randomArray, salt: goodSalt, iterations: 120120, keyLength: 0)
	_ = try PKCS5.PBKDF2(password: randomArray, salt: goodSalt, iterations: 120120, keyLength: 0)
	_ = try Scrypt(password: randomArray, salt: goodSalt, dkLen: 64, N: 16384, r: 8, p: 1)

	// ...
}

```

## References
* [What are Salted Passwords and Password Hashing?](https://www.okta.com/blog/2019/03/what-are-salted-passwords-and-password-hashing/)
* Common Weakness Enumeration: [CWE-760](https://cwe.mitre.org/data/definitions/760.html).
