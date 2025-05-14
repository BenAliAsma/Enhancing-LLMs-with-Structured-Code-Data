# Cleartext storage of sensitive information using a local database on Android
SQLite is a lightweight database engine commonly used in Android devices to store data. By itself, SQLite does not offer any encryption mechanism by default and stores all data in cleartext, which introduces a risk if sensitive data like credentials, authentication tokens or personal identifiable information (PII) are directly stored in a SQLite database. The information could be accessed by any process or user in rooted devices, or can be disclosed through chained vulnerabilities, like unexpected access to the private storage through exposed components.


## Recommendation
Use `SQLCipher` or similar libraries to add encryption capabilities to SQLite. Alternatively, encrypt sensitive data using cryptographically secure algorithms before storing it in the database.


## Example
In the first example, sensitive user information is stored in cleartext.

In the second and third examples, the code encrypts sensitive information before saving it to the database.


```java
public void sqliteStorageUnsafe(Context ctx, String name, String password) {
	// BAD - sensitive information saved in cleartext.
	SQLiteDatabase db = ctx.openOrCreateDatabase("test", Context.MODE_PRIVATE, null);
	db.execSQL("INSERT INTO users VALUES (?, ?)", new String[] {name, password});
}

public void sqliteStorageSafe(Context ctx, String name, String password) {
	// GOOD - sensitive information encrypted with a custom method.
	SQLiteDatabase db = ctx.openOrCreateDatabase("test", Context.MODE_PRIVATE, null);
	db.execSQL("INSERT INTO users VALUES (?, ?)", new String[] {name, encrypt(password)});
}

public void sqlCipherStorageSafe(String name, String password, String databasePassword) {
	// GOOD - sensitive information saved using SQLCipher.
	net.sqlcipher.database.SQLiteDatabase db = 
		net.sqlcipher.database.SQLiteDatabase.openOrCreateDatabase("test", databasePassword, null);
	db.execSQL("INSERT INTO users VALUES (?, ?)", new String[] {name, password});
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
* SQLCipher: [Android Application Integration](https://www.zetetic.net/sqlcipher/sqlcipher-for-android/)
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
