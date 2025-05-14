# Insecure randomness
If you use a cryptographically weak pseudo-random number generator to generate security-sensitive values, such as passwords, attackers can more easily predict those values.

Pseudo-random number generators generate a sequence of numbers that only approximates the properties of random numbers. The sequence is not truly random because it is completely determined by a relatively small set of initial values (the seed). If the random number generator is cryptographically weak, then this sequence may be easily predictable through outside observations.


## Recommendation
The `java.util.Random` random number generator is not cryptographically secure. Use a secure random number generator such as `java.security.SecureRandom` instead.

Use a cryptographically secure pseudo-random number generator if the output is to be used in a security-sensitive context. As a general rule, a value should be considered "security-sensitive" if predicting it would allow the attacker to perform an action that they would otherwise be unable to perform. For example, if an attacker could predict the random password generated for a new user, they would be able to log in as that new user.


## Example
The following examples show different ways of generating a cookie with a random value.

In the first (BAD) case, we generate a fresh cookie by appending a random integer to the end of a static string. The random number generator used (`Random`) is not cryptographically secure, so it may be possible for an attacker to predict the generated cookie.


```java
Random r = new Random();

byte[] bytes = new byte[16];
r.nextBytes(bytes);

String cookieValue = encode(bytes);

Cookie cookie = new Cookie("name", cookieValue);
response.addCookie(cookie);

```
In the second (GOOD) case, we generate a fresh cookie by appending a random integer to the end of a static string. The random number generator used (`SecureRandom`) is cryptographically secure, so it is not possible for an attacker to predict the generated cookie.


```java
SecureRandom r = new SecureRandom();

byte[] bytes = new byte[16];
r.nextBytes(bytes);

String cookieValue = encode(bytes);

Cookie cookie = new Cookie("name", cookieValue);
response.addCookie(cookie);

```

## References
* Wikipedia: [Pseudo-random number generator](http://en.wikipedia.org/wiki/Pseudorandom_number_generator).
* Java Docs: [Random](http://docs.oracle.com/javase/8/docs/api/java/util/Random.html).
* Java Docs: [SecureRandom](http://docs.oracle.com/javase/8/docs/api/java/security/SecureRandom.html).
* Common Weakness Enumeration: [CWE-330](https://cwe.mitre.org/data/definitions/330.html).
* Common Weakness Enumeration: [CWE-338](https://cwe.mitre.org/data/definitions/338.html).
