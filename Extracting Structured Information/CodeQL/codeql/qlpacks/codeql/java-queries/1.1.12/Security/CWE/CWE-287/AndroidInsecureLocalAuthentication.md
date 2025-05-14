# Insecure local authentication
Biometric local authentication such as fingerprint recognition can be used to protect sensitive data or actions within an application. However, if this authentication does not use a `KeyStore`-backed key, it can be bypassed by a privileged malicious application, or by an attacker with physical access using application hooking tools such as Frida.


## Recommendation
Generate a secure key in the Android `KeyStore`. Ensure that the `onAuthenticationSuccess` callback for a biometric prompt uses it in a way that is required for the sensitive parts of the application to function, such as by using it to decrypt sensitive data or credentials.


## Example
In the following (bad) case, no `CryptoObject` is required for the biometric prompt to grant access, so it can be bypassed.


```java
biometricPrompt.authenticate(
    cancellationSignal,
    executor,
    new BiometricPrompt.AuthenticationCallback {
        @Override
        // BAD: This authentication callback does not make use of a `CryptoObject` from the `result`.
        public void onAuthenticationSucceeded(BiometricPrompt.AuthenticationResult result) {
            grantAccess()
        }
    }
)
```
In the following (good) case, a secret key is generated in the Android `KeyStore`. The application requires this secret key for access, using it to decrypt data.


```java
private void generateSecretKey() {
    KeyGenParameterSpec keyGenParameterSpec = new KeyGenParameterSpec.Builder(
        "MySecretKey",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
        .setBlockModes(KeyProperties.BLOCK_MODE_CBC)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_PKCS7)
        .setUserAuthenticationRequired(true)
        .setInvalidatedByBiometricEnrollment(true)
        .build();
    KeyGenerator keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
    keyGenerator.init(keyGenParameterSpec);
    keyGenerator.generateKey();
}


private SecretKey getSecretKey() {
    KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
    keyStore.load(null);
    return ((SecretKey)keyStore.getKey("MySecretKey", null));
}

private Cipher getCipher() {
    return Cipher.getInstance(KeyProperties.KEY_ALGORITHM_AES + "/"
            + KeyProperties.BLOCK_MODE_CBC + "/"
            + KeyProperties.ENCRYPTION_PADDING_PKCS7);
}

public prompt(byte[] encryptedData) {
    Cipher cipher = getCipher();
    SecretKey secretKey = getSecretKey();
    cipher.init(Cipher.DECRYPT_MODE, secretKey);

    biometricPrompt.authenticate(
        new BiometricPrompt.CryptoObject(cipher),
        cancellationSignal,
        executor,
        new BiometricPrompt.AuthenticationCallback() {
            @Override
            // GOOD: This authentication callback uses the result to decrypt some data.
            public void onAuthenticationSucceeded(BiometricPrompt.AuthenticationResult result) {
                Cipher cipher = result.getCryptoObject().getCipher();
                byte[] decryptedData = cipher.doFinal(encryptedData);
                grantAccessWithData(decryptedData);
            }
        }
    );
}
```

## References
* OWASP Mobile Application Security: [Android Local Authentication](https://mas.owasp.org/MASTG/Android/0x05f-Testing-Local-Authentication/)
* OWASP Mobile Application Security: [Testing Biometric Authentication](https://mas.owasp.org/MASTG/tests/android/MASVS-AUTH/MASTG-TEST-0018/)
* WithSecure: [How Secure is your Android Keystore Authentication?](https://labs.withsecure.com/publications/how-secure-is-your-android-keystore-authentication)
* Android Developers: [Biometric Authentication](https://developer.android.com/training/sign-in/biometric-auth)
* Common Weakness Enumeration: [CWE-287](https://cwe.mitre.org/data/definitions/287.html).
