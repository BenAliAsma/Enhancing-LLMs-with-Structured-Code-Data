# Header checking disabled
This rule finds places in the code where header checking is disabled. When header checking is enabled, which is the default, the `\r` or `\n` characters found in a response header are encoded to `%0d` and `%0a`. This defeats header-injection attacks by making the injected material part of the same header line. If you disable header checking, you open potential attack vectors against your client code.


## Recommendation
Do not disable header checking.


## References
* MSDN. [HttpRuntimeSection.EnableHeaderChecking Property](http://msdn.microsoft.com/en-us/library/system.web.configuration.httpruntimesection.enableheaderchecking.aspx).
* Common Weakness Enumeration: [CWE-113](https://cwe.mitre.org/data/definitions/113.html).
