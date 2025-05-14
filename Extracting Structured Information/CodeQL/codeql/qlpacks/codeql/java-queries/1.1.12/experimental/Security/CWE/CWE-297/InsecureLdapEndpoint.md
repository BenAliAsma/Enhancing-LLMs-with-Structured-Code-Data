# Insecure LDAPS Endpoint Configuration
Java versions 8u181 or greater have enabled LDAPS endpoint identification by default. Nowadays infrastructure services like LDAP are commonly deployed behind load balancers therefore the LDAP server name can be different from the FQDN of the LDAPS endpoint. If a service certificate does not properly contain a matching DNS name as part of the certificate, Java will reject it by default.

Instead of addressing the issue properly by having a compliant certificate deployed, frequently developers simply disable the LDAPS endpoint check.

Failing to validate the certificate makes the SSL session susceptible to a man-in-the-middle attack. This query checks whether the LDAPS endpoint check is disabled in system properties.


## Recommendation
Replace any non-conforming LDAP server certificates to include a DNS name in the subjectAltName field of the certificate that matches the FQDN of the service.


## Example
The following two examples show two ways of configuring LDAPS endpoint. In the 'BAD' case, endpoint check is disabled. In the 'GOOD' case, endpoint check is left enabled through the default Java configuration.


```java
public class InsecureLdapEndpoint {
    public Hashtable<String, String> createConnectionEnv() {
        Hashtable<String, String> env = new Hashtable<String, String>();
        env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
        env.put(Context.PROVIDER_URL, "ldaps://ad.your-server.com:636");

        env.put(Context.SECURITY_AUTHENTICATION, "simple");
        env.put(Context.SECURITY_PRINCIPAL, "username");
        env.put(Context.SECURITY_CREDENTIALS, "secpassword");

        // BAD - Test configuration with disabled SSL endpoint check.
        {
            System.setProperty("com.sun.jndi.ldap.object.disableEndpointIdentification", "true");
        }

        return env;
    }
}

```

```java
public class InsecureLdapEndpoint2 {
    public Hashtable<String, String> createConnectionEnv() {
        Hashtable<String, String> env = new Hashtable<String, String>();
        env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
        env.put(Context.PROVIDER_URL, "ldaps://ad.your-server.com:636");

        env.put(Context.SECURITY_AUTHENTICATION, "simple");
        env.put(Context.SECURITY_PRINCIPAL, "username");
        env.put(Context.SECURITY_CREDENTIALS, "secpassword");

        // GOOD - No configuration to disable SSL endpoint check since it is enabled by default.
        {
        }

        return env;
    }
}

```

## References
* Oracle Java 8 Update 181 (8u181): [Endpoint identification enabled on LDAPS connections](https://www.java.com/en/download/help/release_changes.html)
* IBM: [Fix this LDAP SSL error](https://www.ibm.com/support/pages/how-do-i-fix-ldap-ssl-error-%E2%80%9Cjavasecuritycertcertificateexception-no-subject-alternative-names-present%E2%80%9D-websphere-application-server)
* Common Weakness Enumeration: [CWE-297](https://cwe.mitre.org/data/definitions/297.html).
