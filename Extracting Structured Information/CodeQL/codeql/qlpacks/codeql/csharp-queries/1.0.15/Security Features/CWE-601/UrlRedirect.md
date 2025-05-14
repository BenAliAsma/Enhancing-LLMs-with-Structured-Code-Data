# URL redirection from remote source
Directly incorporating user input into a URL redirect request without validating the input can facilitate phishing attacks. In these attacks, unsuspecting users can be redirected to a malicious site that looks very similar to the real site they intend to visit, but which is controlled by the attacker.


## Recommendation
To guard against untrusted URL redirection, it is advisable to avoid putting user input directly into a redirect URL. Instead, maintain a list of authorized redirects on the server; then choose from that list based on the user input provided.

If this is not possible, then the user input should be validated in some other way, for example, by verifying that the target URL is on the same host as the current page.


## Example
The following example shows an HTTP request parameter being used directly in a URL redirect without validating the input, which facilitates phishing attacks:


```csharp
using System;
using System.Web;

public class UnvalidatedUrlHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        // BAD: a request parameter is incorporated without validation into a URL redirect
        ctx.Response.Redirect(ctx.Request.QueryString["page"]);
    }
}

```
One way to remedy the problem is to validate the user input against a known fixed string before doing the redirection:


```csharp
using System;
using System.Web;
using System.Collections.Generic;

public class UnvalidatedUrlHandler : IHttpHandler
{
    private List<string> VALID_REDIRECTS = new List<string>{ "http://cwe.mitre.org/data/definitions/601.html", "http://cwe.mitre.org/data/definitions/79.html" };

    public void ProcessRequest(HttpContext ctx)
    {
        if (VALID_REDIRECTS.Contains(ctx.Request.QueryString["page"]))
        {
            // GOOD: the request parameter is validated against a known list of strings
            ctx.Response.Redirect(ctx.Request.QueryString["page"]);
        }
    }
}
```
Alternatively, we can check that the target URL does not redirect to a different host by checking that the URL is either relative or on a known good host:


```csharp
using System;
using System.Web;

public class UnvalidatedUrlHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        var urlString = ctx.Request.QueryString["page"];
        var url = new Uri(urlString, UriKind.RelativeOrAbsolute);

        var url = new Uri(redirectUrl, UriKind.RelativeOrAbsolute);
        if (!url.IsAbsoluteUri) {
            // GOOD: The redirect is to a relative URL
            ctx.Response.Redirect(url.ToString());
        }

        if (url.Host == "example.org") {
            // GOOD: The redirect is to a known host
            ctx.Response.Redirect(url.ToString());
        }
    }
}
```
Note that as written, the above code will allow redirects to URLs on `example.com`, which is harmless but perhaps not intended. You can substitute your own domain (if known) for `example.com` to prevent this.


## References
* OWASP: [ Unvalidated Redirects and Forwards Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html).
* Microsoft Docs: [Preventing Open Redirection Attacks (C\#)](https://docs.microsoft.com/en-us/aspnet/mvc/overview/security/preventing-open-redirection-attacks).
* Common Weakness Enumeration: [CWE-601](https://cwe.mitre.org/data/definitions/601.html).
