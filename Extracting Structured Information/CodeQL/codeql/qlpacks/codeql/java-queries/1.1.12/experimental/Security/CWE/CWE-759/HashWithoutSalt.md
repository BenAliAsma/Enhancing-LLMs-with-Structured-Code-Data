# Use of a hash function without a salt
In cryptography, a salt is some random data used as an additional input to a one-way function that hashes a password or pass-phrase. It makes dictionary attacks more difficult.

Without a salt, it is much easier for attackers to pre-compute the hash value using dictionary attack techniques such as rainbow tables to crack passwords.


## Recommendation
Use a long random salt of at least 32 bytes then use the combination of password and salt to hash a password or password phrase.


## Example
The following example shows two ways of hashing. In the 'BAD' case, no salt is provided. In the 'GOOD' case, a salt is provided.


```java
public class HashWithoutSalt {
	// BAD - Hash without a salt.
	public void getSHA256Hash(String password) throws NoSuchAlgorithmException {
		MessageDigest md = MessageDigest.getInstance("SHA-256");
		byte[] messageDigest = md.digest(password.getBytes());
	}

	// GOOD - Hash with a salt.
	public void getSHA256Hash(String password, byte[] salt) throws NoSuchAlgorithmException {
		MessageDigest md = MessageDigest.getInstance("SHA-256");
		md.update(salt);
		byte[] messageDigest = md.digest(password.getBytes());
	}
}
```

## References
* DZone: [A Look at Java Cryptography](https://dzone.com/articles/a-look-at-java-cryptography)
* CWE: [CWE-759: Use of a One-Way Hash without a Salt](https://cwe.mitre.org/data/definitions/759.html)
* Common Weakness Enumeration: [CWE-759](https://cwe.mitre.org/data/definitions/759.html).
