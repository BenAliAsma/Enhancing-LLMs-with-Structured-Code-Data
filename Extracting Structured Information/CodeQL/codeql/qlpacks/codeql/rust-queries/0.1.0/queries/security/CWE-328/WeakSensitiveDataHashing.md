# Use of a broken or weak cryptographic hashing algorithm on sensitive data
A broken or weak cryptographic hash function can leave data vulnerable, and should not be used in security-related code.

A strong cryptographic hash function should be resistant to:

* **Pre-image attacks**. If you know a hash value `h(x)`, you should not be able to easily find the input `x`.
* **Collision attacks**. If you know a hash value `h(x)`, you should not be able to easily find a different input `y` with the same hash value `h(x) = h(y)`.
* **Brute force**. For passwords and other data with limited input space, if you know a hash value `h(x)`, you should not be able to find the input `x` even using a brute force attack (without significant computational effort).
As an example, both MD5 and SHA-1 are known to be vulnerable to collision attacks.

All of MD5, SHA-1, SHA-2 and SHA-3 are weak against offline brute forcing, so they are not suitable for hashing passwords. This includes SHA-224, SHA-256, SHA-384, and SHA-512, which are in the SHA-2 family.

Since it's OK to use a weak cryptographic hash function in a non-security context, this query only alerts when these are used to hash sensitive data (such as passwords, certificates, usernames).


## Recommendation
Ensure that you use a strong, modern cryptographic hash function, such as:

* Argon2, scrypt, bcrypt, or PBKDF2 for passwords and other data with limited input space where a dictionary-like attack is feasible.
* SHA-2, or SHA-3 in other cases.
Note that special purpose algorithms, which are used to ensure that a message comes from a particular sender, exist for message authentication. These algorithms should be used when appropriate, as they address common vulnerabilities of simple hashing schemes in this context.


## Example
The following examples show hashing sensitive data using the MD5 hashing algorithm that is known to be vulnerable to collision attacks, and hashing passwords using the SHA-3 algorithm that is weak to brute force attacks:


```rust
// MD5 is not appropriate for hashing sensitive data.
let mut md5_hasher = md5::Md5::new();
...
md5_hasher.update(emergency_contact); // BAD
md5_hasher.update(credit_card_no); // BAD
...
my_hash = md5_hasher.finalize();

// SHA3-256 is not appropriate for hashing passwords.
my_hash = sha3::Sha3_256::digest(password); // BAD

```
To make these secure, we can use the SHA-3 algorithm for sensitive data and Argon2 for passwords:


```rust
// SHA3-256 *is* appropriate for hashing sensitive data.
let mut sha3_256_hasher = sha3::Sha3_256::new();
...
sha3_256_hasher.update(emergency_contact); // GOOD
sha3_256_hasher.update(credit_card_no); // GOOD
...
my_hash = sha3_256_hasher.finalize();

// Argon2 is appropriate for hashing passwords.
let argon2_salt = argon2::password_hash::Salt::from_b64(salt)?;
my_hash = argon2::Argon2::default().hash_password(password.as_bytes(), argon2_salt)?.to_string(); // GOOD

```

## References
* OWASP: [ Password Storage Cheat Sheet ](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) and [ Transport Layer Security Cheat Sheet ](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Security_Cheat_Sheet.html).
* GitHub: [ RustCrypto: Hashes ](https://github.com/RustCrypto/hashes?tab=readme-ov-file#rustcrypto-hashes) and [ RustCrypto: Password Hashes ](https://github.com/RustCrypto/password-hashes?tab=readme-ov-file#rustcrypto-password-hashes).
* The RustCrypto Book: [ Password Hashing ](https://rustcrypto.org/key-derivation/hashing-password.html).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
* Common Weakness Enumeration: [CWE-328](https://cwe.mitre.org/data/definitions/328.html).
* Common Weakness Enumeration: [CWE-916](https://cwe.mitre.org/data/definitions/916.html).
