# Use of a cryptographic algorithm with insufficient key size
Using cryptographic algorithms with a small key size can leave data vulnerable to being decrypted.

Many cryptographic algorithms provided by cryptography libraries can be configured with key sizes that are vulnerable to brute force attacks. Using such a key size means that an attacker may be able to easily decrypt the encrypted data.


## Recommendation
Ensure that you use a strong, modern cryptographic algorithm. Use at least AES-128 or RSA-2048.


## Example
The following code shows an example of using the `openssl` library to generate an RSA key. When creating a key, you must specify which key size to use. The first example uses 1024 bits, which is not considered sufficient. The second example uses 2048 bits, which is currently considered sufficient.


```c
void encrypt_with_openssl(EVP_PKEY_CTX *ctx) {

  // BAD: only 1024 bits for an RSA key
  EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 1024);

  // GOOD: 2048 bits for an RSA key
  EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048);
}
```

## References
* NIST, FIPS 140 Annex a: [ Approved Security Functions](http://csrc.nist.gov/publications/fips/fips140-2/fips1402annexa.pdf).
* NIST, SP 800-131A: [ Transitions: Recommendation for Transitioning the Use of Cryptographic Algorithms and Key Lengths](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar2.pdf).
* Common Weakness Enumeration: [CWE-326](https://cwe.mitre.org/data/definitions/326.html).
