# ASP.NET: leftover debug code
Leftover entry points in web applications can be exploited by an attacker to enter a deployed web application through a way that was not intended and probably not even considered. Static ` Main(..)` methods are typical leftover entry points that can prove harmful to your web application.


## Recommendation
Remove debug code if your web application is in production.


## References
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
