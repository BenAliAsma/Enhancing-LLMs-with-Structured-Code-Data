# Timing attack against signature validation
A constant-time algorithm should be used for checking a MAC or a digital signature. In other words, the comparison time should not depend on the content of the input. Otherwise, an attacker may be able to forge a valid signature for an arbitrary message by running a timing attack if they can send to the validation procedure both the message and the signature. A successful attack can result in authentication bypass.


## Recommendation
Use `MessageDigest.isEqual()` method to check MACs and signatures. If this method is used, then the calculation time depends only on the length of input byte arrays, and does not depend on the contents of the arrays.


## Example
The following example uses `Arrays.equals()` method for validating a MAC over a message. This method implements a non-constant-time algorithm. Both the message and the signature come from an untrusted HTTP request:


```java
public boolean validate(HttpRequest request, SecretKey key) throws Exception {
    byte[] message = getMessageFrom(request);
    byte[] signature = getSignatureFrom(request);

    Mac mac = Mac.getInstance("HmacSHA256");
    mac.init(new SecretKeySpec(key.getEncoded(), "HmacSHA256"));
    byte[] actual = mac.doFinal(message);
    return Arrays.equals(signature, actual);
}
```
The next example uses a safe constant-time algorithm for validating a MAC:


```java
public boolean validate(HttpRequest request, SecretKey key) throws Exception {
    byte[] message = getMessageFrom(request);
    byte[] signature = getSignatureFrom(request);

    Mac mac = Mac.getInstance("HmacSHA256");
    mac.init(new SecretKeySpec(key.getEncoded(), "HmacSHA256"));
    byte[] actual = mac.doFinal(message);
    return MessageDigest.isEqual(signature, actual);
}
```

## References
* Wikipedia: [Timing attack](https://en.wikipedia.org/wiki/Timing_attack).
* Coursera: [Timing attacks on MAC verification](https://www.coursera.org/lecture/crypto/timing-attacks-on-mac-verification-FHGW1)
* NCC Group: [Time Trial: Racing Towards Practical Remote Timing Attacks](https://www.nccgroup.trust/globalassets/our-research/us/whitepapers/TimeTrial.pdf)
* Java API Specification: [MessageDigest.isEqual() method](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/security/MessageDigest.html#isEqual(byte[],byte[]))
* Common Weakness Enumeration: [CWE-208](https://cwe.mitre.org/data/definitions/208.html).
