# Creating an ASP.NET debug binary may reveal sensitive information
ASP.NET applications that deploy a 'debug' build to production can reveal debugging information to end users. This debugging information can aid a malicious user in attacking the system. The use of the debugging flag may also impair performance, increasing execution time and memory usage.


## Recommendation
Remove the 'debug' flag from the `Web.config` file if this configuration is likely to be used in production.


## Example
The following example shows the 'debug' flag set to true in a `Web.config` file for ASP.NET:


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <compilation
      defaultLanguage="c#"
      debug="true"
    />
   ...
  </system.web>
</configuration>
```
This will produce a 'debug' build that may be exploited by an end user.

To fix this problem, the 'debug' flag should be set to `false`, or removed completely:


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <compilation
      defaultLanguage="c#"
    />
   ...
  </system.web>
</configuration>
```

## References
* MSDN: [Why debug=false in ASP.NET applications in production environment](https://web.archive.org/web/20190919105353/https://blogs.msdn.microsoft.com/prashant_upadhyay/2011/07/14/why-debugfalse-in-asp-net-applications-in-production-environment/).
* MSDN: [How to: Enable Debugging for ASP.NET Applications](https://msdn.microsoft.com/en-us/library/e8z01xdh.aspx).
* Common Weakness Enumeration: [CWE-11](https://cwe.mitre.org/data/definitions/11.html).
* Common Weakness Enumeration: [CWE-532](https://cwe.mitre.org/data/definitions/532.html).
