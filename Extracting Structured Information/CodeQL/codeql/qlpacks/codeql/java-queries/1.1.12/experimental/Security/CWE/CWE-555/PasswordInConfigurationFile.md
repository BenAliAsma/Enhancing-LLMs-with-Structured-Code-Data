# Password in configuration file
Storing a plaintext password in a configuration file allows anyone who can read the file to access the password-protected resources.


## Recommendation
Passwords stored in configuration files should be encrypted. Utilities provided by application servers like keystore and password vault can be used to encrypt and manage passwords.


## Example
In the first example, the password of a datasource configuration is stored in cleartext in the context.xml file of a Java EE application.

In the second example, the password of a datasource configuration is encrypted and managed by a password vault.


```xml
<?xml version="1.0" encoding="UTF-8"?>
<Context>
    <!-- BAD: Password of datasource is not encrypted -->
    <Resource name="jdbc/exampleDS" auth="Container" type="javax.sql.DataSource"
               maxTotal="100" maxIdle="30" maxWaitMillis="10000"
               username="root" password="1234"
               driverClassName="com.mysql.jdbc.Driver"
               url="jdbc:mysql://www.example.com:3306/proj"/>

    <!-- GOOD: Password is encrypted and stored in a password vault -->
    <Resource name="jdbc/exampleDS" auth="Container" type="javax.sql.DataSource"
               maxTotal="100" maxIdle="30" maxWaitMillis="10000"
               username="root" password="${VAULT::exampleDS::password::N2NhZDYzOTMtNWE0OS00ZGQ0LWE4MmEtMWNlMDMyNDdmNmI2TElORV9CUkVBS3ZhdWx0}"
               driverClassName="com.mysql.jdbc.Driver"
               url="jdbc:mysql://www.example.com:3306/proj"/>

</Context>
```

## References
* CWE: [CWE-555: J2EE Misconfiguration: Plaintext Password in Configuration File](https://cwe.mitre.org/data/definitions/555.html)
* RedHat Security Guide: [Store and Retrieve Encrypted Sensitive Strings in the Java Keystore](https://access.redhat.com/documentation/en-us/jboss_enterprise_application_platform/6.1/html/security_guide/Store_and_Retrieve_Encrypted_Sensitive_Strings_in_the_Java_Keystore)
* SonarSource: [Hard-coded credentials are security-sensitive](https://rules.sonarsource.com/java/RSPEC-2068)
* Common Weakness Enumeration: [CWE-555](https://cwe.mitre.org/data/definitions/555.html).
* Common Weakness Enumeration: [CWE-256](https://cwe.mitre.org/data/definitions/256.html).
* Common Weakness Enumeration: [CWE-260](https://cwe.mitre.org/data/definitions/260.html).
