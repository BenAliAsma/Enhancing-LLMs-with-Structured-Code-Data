# Use of HTMLInputHidden
Data entered in hidden fields is cached in the same way as the rest of the page, and can be accessed or replaced by attackers that have access to the browser's cache. You should not trust the contents of hidden fields more than the contents of normal input fields.


## Recommendation
Ensure no sensitive information is stored in hidden fields.


## References
* Common Weakness Enumeration: [CWE-472](https://cwe.mitre.org/data/definitions/472.html).
