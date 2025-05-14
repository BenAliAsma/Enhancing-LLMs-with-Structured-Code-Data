# Android missing certificate pinning
Certificate pinning is the practice of only trusting a specific set of SSL certificates, rather than those that the device trusts by default. In Android applications, it is reccomended to use certificate pinning when communicating over the network, in order to minimize the risk of machine-in-the-middle attacks from a compromised CA.


## Recommendation
The easiest way to implement certificate pinning is to declare your pins in a `network-security-config` XML file. This will automatically provide certificate pinning for any network connection made by the app.

Another way to implement certificate pinning is to use the \`CertificatePinner\` class from the \`okhttp\` library.

A final way to implement certificate pinning is to use a `TrustManager`, initialized from a `KeyStore` loaded with only the necessary certificates.


## Example
In the first (bad) case below, a network call is performed with no certificate pinning implemented. The other (good) cases demonstrate the different ways to implement certificate pinning.


```java
// BAD - By default, this network call does not use certificate pinning
URLConnection conn = new URL("https://example.com").openConnection();
```

```xml
<!-- GOOD: Certificate pinning implemented via a Network Security Config file -->

<!-- In AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.app">

    <application android:networkSecurityConfig="@xml/NetworkSecurityConfig">
        ...
    </application>

</manifest>

<!-- In res/xml/NetworkSecurityConfig.xml -->
<network-security-config>
    <domain-config>
        <domain>good.example.com</domain>
        <pin-set expiration="2038/1/19">
            <pin digest="SHA-256">...</pin>
        </pin-set>
    </domain-config>
</network-security-config>
```

```java
// GOOD: Certificate pinning implemented via okhttp3.CertificatePinner 
CertificatePinner certificatePinner = new CertificatePinner.Builder()
    .add("example.com", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    .build();
OkHttpClient client = new OkHttpClient.Builder()
    .certificatePinner(certificatePinner)
    .build();

client.newCall(new Request.Builder().url("https://example.com").build()).execute();



// GOOD: Certificate pinning implemented via a TrustManager
KeyStore keyStore = KeyStore.getInstance("BKS");
keyStore.load(resources.openRawResource(R.raw.cert), null);

TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
tmf.init(keyStore);

SSLContext sslContext = SSLContext.getInstance("TLS");
sslContext.init(null, tmf.getTrustManagers(), null);

URL url = new URL("http://www.example.com/");
HttpsURLConnection urlConnection = (HttpsURLConnection) url.openConnection(); 

urlConnection.setSSLSocketFactory(sslContext.getSocketFactory());
```

## References
* OWASP Mobile Security: [Testing Custom Certificate Stores and Certificate Pinning (MSTG-NETWORK-4)](https://mobile-security.gitbook.io/mobile-security-testing-guide/android-testing-guide/0x05g-testing-network-communication#testing-custom-certificate-stores-and-certificate-pinning-mstg-network-4).
* Android Developers: [Network security configuration](https://developer.android.com/training/articles/security-config).
* OkHttp: [CertificatePinner](https://square.github.io/okhttp/4.x/okhttp/okhttp3/-certificate-pinner/).
* Common Weakness Enumeration: [CWE-295](https://cwe.mitre.org/data/definitions/295.html).
