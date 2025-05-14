# Insecure LDAP authentication
When using the Java LDAP API to perform LDAPv3-style extended operations and controls, a context with connection properties including user credentials is started. Transmission of LDAP credentials in cleartext allows remote attackers to obtain sensitive information by sniffing the network.


## Recommendation
Use the `ldaps://` protocol to send credentials through SSL or use SASL authentication.


## Example
In the following (bad) example, a `ldap://` URL is used and credentials will be sent in plaintext.


```java
String ldapUrl = "ldap://ad.your-server.com:389";
Hashtable<String, String> environment = new Hashtable<String, String>();
environment.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
environment.put(Context.PROVIDER_URL, ldapUrl);
environment.put(Context.REFERRAL, "follow");
environment.put(Context.SECURITY_AUTHENTICATION, "simple");
environment.put(Context.SECURITY_PRINCIPAL, ldapUserName);
environment.put(Context.SECURITY_CREDENTIALS, password);
DirContext dirContext = new InitialDirContext(environment);

```
In the following (good) example, a `ldaps://` URL is used so credentials will be encrypted with SSL.


```java
String ldapUrl = "ldaps://ad.your-server.com:636";
Hashtable<String, String> environment = new Hashtable<String, String>();
environment.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
environment.put(Context.PROVIDER_URL, ldapUrl);
environment.put(Context.REFERRAL, "follow");
environment.put(Context.SECURITY_AUTHENTICATION, "simple");
environment.put(Context.SECURITY_PRINCIPAL, ldapUserName);
environment.put(Context.SECURITY_CREDENTIALS, password);
DirContext dirContext = new InitialDirContext(environment);

```
In the following (good) example, a `ldap://` URL is used, but SASL authentication is enabled so that the credentials will be encrypted.


```java
String ldapUrl = "ldap://ad.your-server.com:389";
Hashtable<String, String> environment = new Hashtable<String, String>();
environment.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
environment.put(Context.PROVIDER_URL, ldapUrl);
environment.put(Context.REFERRAL, "follow");
environment.put(Context.SECURITY_AUTHENTICATION, "DIGEST-MD5 GSSAPI");
environment.put(Context.SECURITY_PRINCIPAL, ldapUserName);
environment.put(Context.SECURITY_CREDENTIALS, password);
DirContext dirContext = new InitialDirContext(environment);

```

## References
* Oracle: [LDAP and LDAPS URLs](https://docs.oracle.com/javase/jndi/tutorial/ldap/misc/url.html)
* Oracle: [Simple authentication](https://docs.oracle.com/javase/tutorial/jndi/ldap/simple.html)
* Common Weakness Enumeration: [CWE-522](https://cwe.mitre.org/data/definitions/522.html).
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
