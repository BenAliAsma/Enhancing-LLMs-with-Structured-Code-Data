# Cleartext logging of sensitive information
Attackers could gain access to sensitive information that is logged unencrypted.


## Recommendation
Always make sure to encrypt or obfuscate sensitive information before you log it.

Generally, you should decrypt sensitive information only at the point where it is necessary for it to be used in cleartext.

Be aware that external processes often store the standard output and standard error streams of the application. This will include logged sensitive information.


## Example
The following example code logs user credentials (in this case, their password) in plaintext:


```swift
let password = "P@ssw0rd"
NSLog("User password changed to \(password)")

```
Instead, you should encrypt or obfuscate the credentials, or omit them entirely:


```swift
let password = "P@ssw0rd"
NSLog("User password changed")

```

## References
* M. Dowd, J. McDonald and J. Schuhm, *The Art of Software Security Assessment*, 1st Edition, Chapter 2 - 'Common Vulnerabilities of Encryption', p. 43. Addison Wesley, 2006.
* M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd Edition, Chapter 9 - 'Protecting Secret Data', p. 299. Microsoft, 2002.
* OWASP: [Password Plaintext Storage](https://www.owasp.org/index.php/Password_Plaintext_Storage).
* Common Weakness Enumeration: [CWE-312](https://cwe.mitre.org/data/definitions/312.html).
* Common Weakness Enumeration: [CWE-359](https://cwe.mitre.org/data/definitions/359.html).
* Common Weakness Enumeration: [CWE-532](https://cwe.mitre.org/data/definitions/532.html).
