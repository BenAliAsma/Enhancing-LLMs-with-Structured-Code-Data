# Use of RSA algorithm without OAEP
Cryptographic algorithms often use padding schemes to make the plaintext less predictable. The OAEP (Optimal Asymmetric Encryption Padding) scheme should be used with RSA encryption. Using an outdated padding scheme such as PKCS1, or no padding at all, can weaken the encryption by making it vulnerable to a padding oracle attack.


## Recommendation
Use the OAEP scheme when using RSA encryption.


## Example
In the following example, the BAD case shows no padding being used, whereas the GOOD case shows an OAEP scheme being used.


```java
// BAD: No padding scheme is used
Cipher rsa = Cipher.getInstance("RSA/ECB/NoPadding");
...

//GOOD: OAEP padding is used
Cipher rsa = Cipher.getInstance("RSA/ECB/OAEPWithSHA-1AndMGF1Padding");
...
```

## References
* [Mobile Security Testing Guide](https://github.com/MobSF/owasp-mstg/blob/master/Document/0x04g-Testing-Cryptography.md#padding-oracle-attacks-due-to-weaker-padding-or-block-operation-implementations).
* [The Padding Oracle Attack](https://robertheaton.com/2013/07/29/padding-oracle-attack/).
* Common Weakness Enumeration: [CWE-780](https://cwe.mitre.org/data/definitions/780.html).
