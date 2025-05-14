# Untrusted data passed to external API
Using unsanitized untrusted data in an external API can cause a variety of security issues. This query reports external APIs that use untrusted data. The results are not filtered so that you can audit all examples. The query provides data for security reviews of the application and you can also use it to identify external APIs that should be modeled as either taint steps, or sinks for specific problems.

An external API is defined as a call to a callable (method/property accessor/operator/...) that is not defined in the source code, not overridden in the source code, and is not modeled as a taint step in the default taint library. External APIs may be from the .NET standard library, third-party dependencies or from internal dependencies. The query reports uses of untrusted data in either the qualifier or as one of the arguments of external APIs.


## Recommendation
For each result:

* If the result highlights a known sink, confirm that the result is reported by the relevant query, or that the result is a false positive because this data is sanitized.
* If the result highlights an unknown sink for a problem, then add modeling for the sink to the relevant query, and confirm that the result is either found, or is safe due to appropriate sanitization.
* If the result represents a call to an external API that transfers taint, add the appropriate modeling, and re-run the query to determine what new results have appeared due to this additional modeling.
Otherwise, the result is likely uninteresting. Custom versions of this query can extend the `SafeExternalAPICallable` class to exclude known safe external APIs from future analysis.


## Example
A query string parameter is read from `HttpRequest` and then ultimately used in a call to the `HttpResponse.Write` external API:


```csharp
using System;
using System.Text;
using System.Web;

public class UntrustedData : IHttpHandler
{
    public void ProcessRequest(HttpContext ctx)
    {
        var name = ctx.Request.QueryString["name"];
        ctx.Response.Write(name);
    }

    public bool IsReusable => true;
}

```
This is an XSS sink. The XSS query should therefore be reviewed to confirm that this sink is appropriately modeled, and if it is, to confirm that the query reports this particular result, or that the result is a false positive due to some existing sanitization.


## References
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
