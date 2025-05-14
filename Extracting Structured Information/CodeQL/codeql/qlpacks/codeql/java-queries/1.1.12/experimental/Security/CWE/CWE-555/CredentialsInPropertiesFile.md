# Cleartext Credentials in Properties File
Credentials management issues occur when credentials are stored in plaintext in an application's properties file. Common credentials include but are not limited to LDAP, mail, database, proxy account, and so on. Storing plaintext credentials in a properties file allows anyone who can read the file access to the protected resource. Good credentials management guidelines require that credentials never be stored in plaintext.


## Recommendation
Credentials stored in properties files should be encrypted and recycled regularly. In a Java EE deployment scenario, utilities provided by application servers like keystores and password vaults can be used to encrypt and manage credentials.


## Example
In the first example, the credentials for the LDAP and datasource properties are stored in cleartext in the properties file.

In the second example, the credentials for the LDAP and datasource properties are stored in an encrypted format.


```none
#***************************** LDAP Credentials *****************************************#
 
ldap.ldapHost = ldap.example.com
ldap.ldapPort = 636
ldap.loginDN = cn=Directory Manager

#### BAD: LDAP credentials are stored in cleartext #### 
ldap.password = mysecpass

#### GOOD: LDAP credentials are stored in the encrypted format #### 
ldap.password = eFRZ3Cqo5zDJWMYLiaEupw==

ldap.domain1 = example
ldap.domain2 = com
ldap.url= ldaps://ldap.example.com:636/dc=example,dc=com

#*************************** MS SQL Database Connection **********************************# 
datasource1.driverClassName = com.microsoft.sqlserver.jdbc.SQLServerDriver
datasource1.url = jdbc:sqlserver://ms.example.com\\exampledb:1433;
datasource1.username = sa

#### BAD: Datasource credentials are stored in cleartext #### 
datasource1.password = Passw0rd@123

#### GOOD: Datasource credentials are stored in the encrypted format #### 
datasource1.password = VvOgflYS1EUzJdVNDoBcnA==

```

## References
* OWASP: [Password Plaintext Storage](https://owasp.org/www-community/vulnerabilities/Password_Plaintext_Storage)
* Medium (Rajeev Shukla): [Encrypting database password in the application.properties file](https://medium.com/@mail2rajeevshukla/hiding-encrypting-database-password-in-the-application-properties-34d59fe104eb)
* Common Weakness Enumeration: [CWE-555](https://cwe.mitre.org/data/definitions/555.html).
* Common Weakness Enumeration: [CWE-256](https://cwe.mitre.org/data/definitions/256.html).
* Common Weakness Enumeration: [CWE-260](https://cwe.mitre.org/data/definitions/260.html).
