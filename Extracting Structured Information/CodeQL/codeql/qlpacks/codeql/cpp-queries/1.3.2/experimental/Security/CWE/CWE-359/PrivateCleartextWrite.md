# Exposure of private information
Private data that is stored in a file or buffer unencrypted is accessible to an attacker who gains access to the storage.


## Recommendation
Ensure that private data is always encrypted before being stored. It may be wise to encrypt information before it is put into a buffer that may be readable in memory.

In general, decrypt private data only at the point where it is necessary for it to be used in cleartext.


## References
* [OWASP Sensitive_Data_Exposure](https://owasp.org/www-project-top-ten/OWASP_Top_Ten_2017/Top_10-2017_A3-Sensitive_Data_Exposure)
* M. Dowd, J. McDonald and J. Schuhm, *The Art of Software Security Assessment*, 1st Edition, Chapter 2 - 'Common Vulnerabilities of Encryption', p. 43. Addison Wesley, 2006.
* M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd Edition, Chapter 9 - 'Protecting Secret Data', p. 299. Microsoft, 2002.
* Common Weakness Enumeration: [CWE-359](https://cwe.mitre.org/data/definitions/359.html).
