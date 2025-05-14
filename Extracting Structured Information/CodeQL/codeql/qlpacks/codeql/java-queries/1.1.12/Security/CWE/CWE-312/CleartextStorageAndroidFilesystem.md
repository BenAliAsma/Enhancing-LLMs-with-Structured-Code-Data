# Cleartext storage of sensitive information in the Android filesystem
Android applications with the appropriate permissions can write files either to the device external storage or the application internal storage, depending on the application's needs. However, sensitive information should not be saved in cleartext. Otherwise it can be accessed by any process or user in rooted devices, or can be disclosed through chained vulnerabilities, like unexpected access to the private storage through exposed components.


## Recommendation
Consider using the `EncryptedFile` class to work with files containing sensitive data. Alternatively, use encryption algorithms to encrypt the sensitive data being stored.


## Example
In the first example, sensitive user information is stored in cleartext using a local file.

In the second and third examples, the code encrypts sensitive information before saving it to the filesystem.


```java
public void fileSystemStorageUnsafe(String name, String password) {
	// BAD - sensitive data stored in cleartext
    FileWriter fw = new FileWriter("some_file.txt");
    fw.write(name + ":" + password);
    fw.close();
}

public void filesystemStorageEncryptedFileSafe(Context context, String name, String password) {
	// GOOD - the whole file is encrypted with androidx.security.crypto.EncryptedFile
    File file = new File("some_file.txt");
    String masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC);
    EncryptedFile encryptedFile = new EncryptedFile.Builder(
        file,
        context,
        masterKeyAlias,
        EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB
    ).build();
	FileOutputStream encryptedOutputStream = encryptedFile.openFileOutput();
	encryptedOutputStream.write(name + ":" + password);
}

public void fileSystemStorageSafe(String name, String password) {
	// GOOD - sensitive data is encrypted using a custom method
    FileWriter fw = new FileWriter("some_file.txt");
    fw.write(name + ":" + encrypt(password));
    fw.close();
}

private static String encrypt(String cleartext) {
    // Use an encryption or strong hashing algorithm in the real world.
    // The example below just returns a SHA-256 hash.
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    byte[] hash = digest.digest(cleartext.getBytes(StandardCharsets.UTF_8));
    String encoded = Base64.getEncoder().encodeToString(hash);
    return encoded;
}

```

## References
* Android Developers: [Work with data more securely](https://developer.android.com/topic/security/data)
* Android Developers: [EncryptedFile](https://developer.android.com/reference/androidx/security/crypto/EncryptedFile)
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
