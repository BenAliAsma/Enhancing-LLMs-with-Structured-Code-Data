# Large 'maxRequestLength' value
The `maxRequestLength` attribute sets the limit for the input stream buffering threshold in KB. Attackers can use large requests to cause denial-of-service attacks.


## Recommendation
The recommended value is 4096 KB but you should try setting it as small as possible according to business requirements.


## Example
The following example shows the `maxRequestLength` attribute set to a high value (255 MB) in a `Web.config` file for ASP.NET:


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <httpRuntime maxRequestLength="255000" />
  </system.web>
</configuration>
```
Unless such a high value is strictly needed, it is better to set the recommended value (4096 KB):


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <httpRuntime maxRequestLength="4096" />
  </system.web>
</configuration>
```

## References
* MSDN: [HttpRuntimeSection.MaxRequestLength Property](https://docs.microsoft.com/en-us/dotnet/api/system.web.configuration.httpruntimesection.maxrequestlength?view=netframework-4.8).
* Common Weakness Enumeration: [CWE-16](https://cwe.mitre.org/data/definitions/16.html).
