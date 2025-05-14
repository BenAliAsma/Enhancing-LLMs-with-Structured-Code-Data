# Use of a broken or weak cryptographic algorithm
Using broken or weak cryptographic algorithms can leave data vulnerable to being decrypted or forged by an attacker.

Many cryptographic algorithms provided by cryptography libraries are known to be weak, or flawed. Using such an algorithm means that encrypted or hashed data is less secure than it appears to be.


## Recommendation
Ensure that you use a strong, modern cryptographic algorithm, such as AES-128 or RSA-2048.


## Example
The following code uses the `OpenSSL` library to encrypt some secret data. When you create a cipher using `OpenSSL` you must specify the encryption algorithm to use. The first example uses DES, which is an older algorithm that is now considered weak. The second example uses AES, which is a stronger modern algorithm.


```ruby
require 'openssl'

class Encryptor
  attr_accessor :secret_key

  def encrypt_message_weak(message)
    cipher = OpenSSL::Cipher.new('des') # BAD: weak encryption
    cipher.encrypt
    cipher.key = secret_key
    cipher.update(message)
    cipher.final
  end

  def encrypt_message_strong(message)
    cipher = OpenSSL::Cipher::AES128.new # GOOD: strong encryption
    cipher.encrypt
    cipher.key = secret_key
    cipher.update(message)
    cipher.final
  end
end

```

## References
* NIST, FIPS 140 Annex a: [ Approved Security Functions](http://csrc.nist.gov/publications/fips/fips140-2/fips1402annexa.pdf).
* NIST, SP 800-131A: [ Transitions: Recommendation for Transitioning the Use of Cryptographic Algorithms and Key Lengths](http://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-131Ar1.pdf).
* OWASP: [Rule - Use strong approved cryptographic algorithms](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html#rule---use-strong-approved-authenticated-encryption).
* Common Weakness Enumeration: [CWE-327](https://cwe.mitre.org/data/definitions/327.html).
