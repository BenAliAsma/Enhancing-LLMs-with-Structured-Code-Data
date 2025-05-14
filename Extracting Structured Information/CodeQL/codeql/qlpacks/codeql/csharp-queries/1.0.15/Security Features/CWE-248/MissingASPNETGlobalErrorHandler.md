# Missing global error handler
`Web.config` files that set the `customErrors` mode to `Off` and do not provide an `Application_Error` method in the `global.asax.cs` file rely on the default error pages, which leak information such as stack traces.


## Recommendation
Set the `customErrors` to `On` to prevent the default error page from being displayed, or to `RemoteOnly` to only show the default error page when the application is accessed locally. Alternatively, provide an implementation of the `Application_Error` method in the `global.asax.cs` page.


## Example
The following example shows a `Web.config` file in which the custom errors mode has been set to `Off`.


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <customErrors mode="Off">
      ...
    </customErrors>
  </system.web>
</configuration>

```
This can be fixed either by specifying a different mode, such as `On`, in the `Web.config` file:


```none
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <system.web>
    <customErrors mode="On">
      ...
    </customErrors>
  </system.web>
</configuration>

```
or by defining an `Application_Error` method in the `global.asax.cs` file:


```csharp
using System;
using System.Web;

namespace WebApp
{
    public class Global : HttpApplication
    {
        void Application_Error(object sender, EventArgs e)
        {
            // Handle errors here
        }
    }
}

```

## References
* Common Weakness Enumeration: [CWE-12](https://cwe.mitre.org/data/definitions/12.html).
* Common Weakness Enumeration: [CWE-248](https://cwe.mitre.org/data/definitions/248.html).
