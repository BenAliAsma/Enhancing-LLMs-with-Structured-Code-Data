# Use of a broken or weak cryptographic algorithm
Using broken or weak cryptographic algorithms can leave data vulnerable to being decrypted or forged by an attacker.

Many cryptographic algorithms provided by cryptography libraries are known to be weak, or flawed. Using such an algorithm means that encrypted or hashed data is less secure than it appears to be.

This query alerts on any use of a weak cryptographic algorithm, that is not a hashing algorithm. Use of broken or weak cryptographic hash functions are handled by the `rust/weak-sensitive-data-hashing` query.


## Recommendation
Ensure that you use a strong, modern cryptographic algorithm, such as AES-128 or RSA-2048.


## Example
The following code uses the `des` crate from the `RustCrypto` family to encrypt some secret data. The DES algorithm is old and considered very weak.


```rust
let des_cipher = cbc::Encryptor::<des::Des>::new(key.into(), iv.into()); // BAD: weak encryption
let encryption_result = des_cipher.encrypt_padded_mut::<des::cipher::block_padding::Pkcs7>(data, data_len);

```
Instead, we should use a strong modern algorithm. In this case, we have selected the 256-bit version of the AES algorithm.


```rust
let aes_cipher = cbc::Encryptor::<aes::Aes256>::new(key.into(), iv.into()); // GOOD: strong encryption
let encryption_result = aes_cipher.encrypt_padded_mut::<aes::cipher::block_padding::Pkcs7>(data, data_len);

```

## References
* NIST, FIPS 140 Annex A: [ Approved Security Functions](http://csrc.nist.gov/publications/fips/fips140-2/fips1402annexa.pdf).
* NIST, SP 800-131A Revision 2: [Transitioning the Use of Cryptographic Algorithms and Key Lengths](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar2.pdf).
* OWASP: [ Cryptographic Storage Cheat Sheet - Algorithms](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html#algorithms).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
