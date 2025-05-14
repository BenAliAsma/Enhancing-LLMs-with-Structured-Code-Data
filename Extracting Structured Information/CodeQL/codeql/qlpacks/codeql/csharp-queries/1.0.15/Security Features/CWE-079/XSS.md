# Cross-site scripting
Directly writing user input (for example, an HTTP request parameter) to a webpage, without properly sanitizing the input first, allows for a cross-site scripting vulnerability.


## Recommendation
To guard against cross-site scripting, consider using a library that provides suitable encoding functionality, such as the `System.Net.WebUtility` class, to sanitize the untrusted input before writing it to the page. For other possible solutions, see the references.


## Example
The following example shows the page parameter being written directly to the server error page, leaving the website vulnerable to cross-site scripting.


```csharp
using System;
using System.Web;

public class XSSHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        ctx.Response.Write(
            "The page \"" + ctx.Request.QueryString["page"] + "\" was not found.");
    }
}

```
Sanitizing the user-controlled data using the `WebUtility.HtmlEncode` method prevents the vulnerability:


```csharp
using System;
using System.Web;
using System.Net;

public class XSSHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        string page = WebUtility.HtmlEncode(ctx.Request.QueryString["page"]);
        ctx.Response.Write(
            "The page \"" + page + "\" was not found.");
    }
}

```

## References
* OWASP: [XSS (Cross Site Scripting) Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html).
* Wikipedia: [Cross-site scripting](http://en.wikipedia.org/wiki/Cross-site_scripting).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
