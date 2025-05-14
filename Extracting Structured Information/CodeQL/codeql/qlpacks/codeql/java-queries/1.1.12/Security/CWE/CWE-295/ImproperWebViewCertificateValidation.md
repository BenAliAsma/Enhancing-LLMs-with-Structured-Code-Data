# Android `WebView` that accepts all certificates
If the `onReceivedSslError` method of an Android `WebViewClient` always calls `proceed` on the given `SslErrorHandler`, it trusts any certificate. This allows an attacker to perform a machine-in-the-middle attack against the application, therefore breaking any security Transport Layer Security (TLS) gives.

An attack might look like this:

1. The vulnerable application connects to `https://example.com`.
1. The attacker intercepts this connection and presents a valid, self-signed certificate for `https://example.com`.
1. The vulnerable application calls the `onReceivedSslError` method to check whether it should trust the certificate.
1. The `onReceivedSslError` method of your `WebViewClient` calls `SslErrorHandler.proceed`.
1. The vulnerable application accepts the certificate and proceeds with the connection since your `WevViewClient` trusted it by proceeding.
1. The attacker can now read the data your application sends to `https://example.com` and/or alter its replies while the application thinks the connection is secure.

## Recommendation
Do not use a call `SslerrorHandler.proceed` unconditionally. If you have to use a self-signed certificate, only accept that certificate, not all certificates.


## Example
In the first (bad) example, the `WebViewClient` trusts all certificates by always calling `SslErrorHandler.proceed`. In the second (good) example, only certificates signed by a certain public key are accepted.


```java
class Bad extends WebViewClient {
    // BAD: All certificates are trusted.
    public void onReceivedSslError (WebView view, SslErrorHandler handler, SslError error) { // $hasResult
        handler.proceed(); 
    }
}

class Good extends WebViewClient {
    PublicKey myPubKey = ...;

    // GOOD: Only certificates signed by a certain public key are trusted.
    public void onReceivedSslError (WebView view, SslErrorHandler handler, SslError error) { // $hasResult
        try {
            X509Certificate cert = error.getCertificate().getX509Certificate();
            cert.verify(this.myPubKey);
            handler.proceed();
        }
        catch (CertificateException|NoSuchAlgorithmException|InvalidKeyException|NoSuchProviderException|SignatureException e) {
            handler.cancel();
        }
    }    
}
```

## References
* [WebViewClient.onReceivedSslError documentation](https://developer.android.com/reference/android/webkit/WebViewClient?hl=en#onReceivedSslError(android.webkit.WebView,%20android.webkit.SslErrorHandler,%20android.net.http.SslError)).
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
