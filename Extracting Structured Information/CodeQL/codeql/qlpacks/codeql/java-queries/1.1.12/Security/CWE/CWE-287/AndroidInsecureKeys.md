# Insecurely generated keys for local authentication
Biometric authentication, such as fingerprint recognition, can be used alongside cryptographic keys stored in the Android `KeyStore` to protect sensitive parts of the application. However, when a key generated for this purpose has certain parameters set insecurely, an attacker with physical access can bypass the authentication check using application hooking tools such as Frida.


## Recommendation
When generating a key for use with biometric authentication, ensure that the following parameters of `KeyGenParameterSpec.Builder` are set:

* `setUserAuthenticationRequired` should be set to `true`; otherwise, the key can be used without user authentication.
* `setInvalidatedByBiometricEnrollment` should be set to `true` (the default); otherwise, an attacker can use the key by enrolling additional biometrics on the device.
* `setUserAuthenticationValidityDurationSeconds`, if used, should be set to `-1`; otherwise, non-biometric (less secure) credentials can be used to access the key. We recommend using `setUserAuthenticationParameters` instead to explicitly set both the timeout and the types of credentials that may be used.

## Example
The following example demonstrates a key that is configured with secure paramaters:


```java
private void generateSecretKey() {
    KeyGenParameterSpec keyGenParameterSpec = new KeyGenParameterSpec.Builder(
        "MySecretKey",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
        // GOOD: Secure parameters are used to generate a key for biometric authentication.
        .setUserAuthenticationRequired(true)
        .setInvalidatedByBiometricEnrollment(true)
        .setUserAuthenticationParameters(0, KeyProperties.AUTH_BIOMETRIC_STRONG)
        .build();
    KeyGenerator keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
    keyGenerator.init(keyGenParameterSpec);
    keyGenerator.generateKey();
}
```
In each of the following cases, a parameter is set insecurely:


```java
private void generateSecretKey() {
    KeyGenParameterSpec keyGenParameterSpec = new KeyGenParameterSpec.Builder(
        "MySecretKey",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
        // BAD: User authentication is not required to use this key.
        .setUserAuthenticationRequired(false)
        .build();
    KeyGenerator keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
    keyGenerator.init(keyGenParameterSpec);
    keyGenerator.generateKey();
}

private void generateSecretKey() {
    KeyGenParameterSpec keyGenParameterSpec = new KeyGenParameterSpec.Builder(
        "MySecretKey",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
        .setUserAuthenticationRequired(true)
        // BAD: An attacker can access this key by enrolling additional biometrics.
        .setInvalidatedByBiometricEnrollment(false)
        .build();
    KeyGenerator keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
    keyGenerator.init(keyGenParameterSpec);
    keyGenerator.generateKey();
}

private void generateSecretKey() {
    KeyGenParameterSpec keyGenParameterSpec = new KeyGenParameterSpec.Builder(
        "MySecretKey",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
        .setUserAuthenticationRequired(true)
        .setInvalidatedByBiometricEnrollment(true)
        // BAD: This key can be accessed using non-biometric credentials. 
        .setUserAuthenticationValidityDurationSeconds(30)
        .build();
    KeyGenerator keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
    keyGenerator.init(keyGenParameterSpec);
    keyGenerator.generateKey();
}
```

## References
* WithSecure: [How Secure is your Android Keystore Authentication?](https://labs.withsecure.com/publications/how-secure-is-your-android-keystore-authentication).
* Android Developers: [KeyGenParameterSpec.Builder](https://developer.android.com/reference/android/security/keystore/KeyGenParameterSpec.Builder).
* Common Weakness Enumeration: [CWE-287](https://cwe.mitre.org/data/definitions/287.html).
