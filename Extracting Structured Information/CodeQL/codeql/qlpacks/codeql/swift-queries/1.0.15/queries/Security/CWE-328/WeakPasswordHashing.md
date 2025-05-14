# Use of an inappropriate cryptographic hashing algorithm on passwords
Hash functions that are not sufficiently computationally hard can leave data vulnerable. You should not use such functions for password hashing.

A strong cryptographic hash function should be resistant to:

* **Pre-image attacks**. If you know a hash value `h(x)`, you should not be able to easily find the input `x`.
* **Collision attacks**. If you know a hash value `h(x)`, you should not be able to easily find a different input `y` with the same hash value `h(x) = h(y)`.
* **Brute force**. If you know a hash value `h(x)`, you should not be able to find an input `y` that computes to that hash value using brute force attacks without significant computational effort.
All of MD5, SHA-1, SHA-2 and SHA-3 are weak against offline brute forcing, since they are not sufficiently computationally hard. This includes SHA-224, SHA-256, SHA-384 and SHA-512, which are in the SHA-2 family.

Password hashing algorithms should be slow and/or memory intensive to compute, to make brute force attacks more difficult.


## Recommendation
For password storage, you should use a sufficiently computationally hard cryptographic hash function, such as one of the following:

* Argon2
* scrypt
* bcrypt
* PBKDF2

## Example
The following examples show two versions of the same function. In both cases, a password is hashed using a cryptographic hashing algorithm. In the first case, the SHA-512 hashing algorithm is used. It is vulnerable to offline brute force attacks:


```swift
let passwordData = Data(passwordString.utf8)
let passwordHash = Crypto.SHA512.hash(data: passwordData) // BAD: SHA-512 is not suitable for password hashing.

// ...

if Crypto.SHA512.hash(data: Data(passwordString.utf8)) == passwordHash {
    // ...
}

```
Here is the same function using Argon2, which is suitable for password hashing:


```swift
import Argon2Swift

let salt = Salt.newSalt()
let result = try! Argon2Swift.hashPasswordString(password: passwordString, salt: salt) // GOOD: Argon2 is suitable for password hashing.
let passwordHash = result.encodedString()

// ...

if try! Argon2Swift.verifyHashString(password: passwordString, hash: passwordHash) {
    // ...
}

```

## References
* OWASP: [Password Storage Cheat Sheet ](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
* GitHub: [CryptoSwift README - Password-Based Key Derivation Function](https://github.com/krzyzanowskim/CryptoSwift/blob/main/README.md#password-based-key-derivation-function)
* libsodium: [libsodium bindings for other languages](https://doc.libsodium.org/bindings_for_other_languages#bindings-programming-languages)
* GitHub: [Argon2Swift](https://github.com/tmthecoder/Argon2Swift)
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
* Common Weakness Enumeration: [CWE-328](https://cwe.mitre.org/data/definitions/328.html).
* Common Weakness Enumeration: [CWE-916](https://cwe.mitre.org/data/definitions/916.html).
