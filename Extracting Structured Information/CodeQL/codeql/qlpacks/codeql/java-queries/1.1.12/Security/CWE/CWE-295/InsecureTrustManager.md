# `TrustManager` that accepts all certificates
If the `checkServerTrusted` method of a `TrustManager` never throws a `CertificateException`, it trusts every certificate. This allows an attacker to perform a machine-in-the-middle attack against the application, therefore breaking any security Transport Layer Security (TLS) gives.

An attack might look like this:

1. The vulnerable program connects to `https://example.com`.
1. The attacker intercepts this connection and presents a valid, self-signed certificate for `https://example.com`.
1. The vulnerable program calls the `checkServerTrusted` method to check whether it should trust the certificate.
1. The `checkServerTrusted` method of your `TrustManager` does not throw a `CertificateException`.
1. The vulnerable program accepts the certificate and proceeds with the connection since your `TrustManager` implicitly trusted it by not throwing an exception.
1. The attacker can now read the data your program sends to `https://example.com` and/or alter its replies while the program thinks the connection is secure.

## Recommendation
Do not use a custom `TrustManager` that trusts any certificate. If you have to use a self-signed certificate, don't trust every certificate, but instead only trust this specific certificate. See below for an example of how to do this.


## Example
In the first (bad) example, the `TrustManager` never throws a `CertificateException` and therefore implicitly trusts any certificate. This allows an attacker to perform a machine-in-the-middle attack. In the second (good) example, the self-signed certificate that should be trusted is loaded into a `KeyStore`. This explicitly defines the certificate as trusted and there is no need to create a custom `TrustManager`.


```java
public static void main(String[] args) throws Exception {
    {
        class InsecureTrustManager implements X509TrustManager {
            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return null;
            }

            @Override
            public void checkServerTrusted(X509Certificate[] chain, String authType) throws CertificateException {
                // BAD: Does not verify the certificate chain, allowing any certificate.
            }

            @Override
            public void checkClientTrusted(X509Certificate[] chain, String authType) throws CertificateException {

            }
        }
        SSLContext context = SSLContext.getInstance("TLS");
        TrustManager[] trustManager = new TrustManager[] { new InsecureTrustManager() };
        context.init(null, trustManager, null);
    }
    {
        SSLContext context = SSLContext.getInstance("TLS");
        File certificateFile = new File("path/to/self-signed-certificate");
        // Create a `KeyStore` with default type
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        // `keyStore` is initially empty
        keyStore.load(null, null);
        X509Certificate generatedCertificate;
        try (InputStream cert = new FileInputStream(certificateFile)) {
            generatedCertificate = (X509Certificate) CertificateFactory.getInstance("X509")
                    .generateCertificate(cert);
        }
        // Add the self-signed certificate to the key store
        keyStore.setCertificateEntry(certificateFile.getName(), generatedCertificate);
        // Get default `TrustManagerFactory`
        TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
        // Use it with our key store that trusts our self-signed certificate
        tmf.init(keyStore);
        TrustManager[] trustManagers = tmf.getTrustManagers();
        context.init(null, trustManagers, null);
        // GOOD, we are not using a custom `TrustManager` but instead have
        // added the self-signed certificate we want to trust to the key
        // store. Note, the `trustManagers` will **only** trust this one
        // certificate.
        
        URL url = new URL("https://self-signed.badssl.com/");
        HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();
        conn.setSSLSocketFactory(context.getSocketFactory());
    }
}

```

## References
* Android Developers: [Security with HTTPS and SSL](https://developer.android.com/training/articles/security-ssl).
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
