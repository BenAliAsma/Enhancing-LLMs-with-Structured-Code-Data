# Disabled certificate revocation checking
Validating a certificate chain includes multiple steps. One of them is checking whether or not certificates in the chain have been revoked. A certificate may be revoked due to multiple reasons. One of the reasons why the certificate authority (CA) may revoke a certificate is that its private key has been compromised. For example, the private key might have been stolen by an adversary. In this case, the adversary may be able to impersonate the owner of the private key. Therefore, trusting a revoked certificate may be dangerous.

The Java Certification Path API provides a revocation checking mechanism that supports both CRL and OCSP. Revocation checking happens while building and validating certificate chains. If at least one of the certificates is revoked, then an exception is thrown. This mechanism is enabled by default. However, it may be disabled by passing `false` to the `PKIXParameters.setRevocationEnabled()` method. If an application doesn't set a custom `PKIXRevocationChecker` via `PKIXParameters.addCertPathChecker()` or `PKIXParameters.setCertPathCheckers()` methods, then revocation checking is not going to happen.


## Recommendation
An application should not disable the default revocation checking mechanism unless it provides a custom revocation checker.


## Example
The following example turns off revocation checking for validating a certificate chain. That should be avoided.


```java
public void validateUnsafe(KeyStore cacerts, CertPath chain) throws Exception {
    CertPathValidator validator = CertPathValidator.getInstance("PKIX");
    PKIXParameters params = new PKIXParameters(cacerts);
    params.setRevocationEnabled(false);
    validator.validate(chain, params);
}
```
The next example uses the default revocation checking mechanism.


```java
public void validate(KeyStore cacerts, CertPath chain) throws Exception {
    CertPathValidator validator = CertPathValidator.getInstance("PKIX");
    PKIXParameters params = new PKIXParameters(cacerts);
    validator.validate(chain, params);
}
```
The third example turns off the default revocation mechanism. However, it registers another revocation checker that uses OCSP to obtain revocation status of certificates.


```java
public void validate(KeyStore cacerts, CertPath certPath) throws Exception {
    CertPathValidator validator = CertPathValidator.getInstance("PKIX");
    PKIXParameters params = new PKIXParameters(cacerts);
    params.setRevocationEnabled(false);
    PKIXRevocationChecker checker = (PKIXRevocationChecker) validator.getRevocationChecker();
    checker.setOcspResponder(OCSP_RESPONDER_URL);
    checker.setOcspResponderCert(OCSP_RESPONDER_CERT);
    params.addCertPathChecker(checker);
    validator.validate(certPath, params);
}
```

## References
* Wikipedia: [Public key certificate](https://en.wikipedia.org/wiki/Public_key_certificate)
* Java SE Documentation: [Java PKI Programmer's Guide](https://docs.oracle.com/javase/8/docs/technotes/guides/security/certpath/CertPathProgGuide.html)
* Java API Specification: [CertPathValidator](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/security/cert/CertPathValidator.html)
* Common Weakness Enumeration: [CWE-299](https://cwe.mitre.org/data/definitions/299.html).
