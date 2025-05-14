# Inappropriate encoding
Using an inappropriate encoding may yield unintended results. For example, using URL encoding for HTML content may result in an incorrectly rendered web page. Using an inappropriate encoding may further pose a security risk when a remote user can control the value being encoded. For example, redirecting to an HTML encoded URL provided by a remote user can facilitate phishing attacks.


## Recommendation
Use the appropriate encoding. For example, use HTML encoding for HTML entities, URL encoding for URLs, etc. If possible, avoid writing custom encoding/escaping functionality, and use predefined functionality such as `HttpUtility.HtmlEncode()` or `SqlParameter`s instead.


## Example
The following example shows two methods where a value is encoded before being written to an HTML page. In the `Bad` method, URL encoding is incorrectly used. In the `Good` method, HTML encoding is correctly used.


```csharp
using System;
using System.Web;
using System.Net;

public class HtmlEncode
{
    public static void Bad(HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        ctx.Response.Write("Hello, " + WebUtility.UrlEncode(user));
    }

    public static void Good(HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        ctx.Response.Write("Hello, " + WebUtility.HtmlEncode(user));
    }
}

```
The following example shows two methods where a value is encoded before being used in a URL redirect. In the `Bad` method, HTML encoding is incorrectly used. In the `Good` method, URL encoding is correctly used.


```csharp
using System;
using System.Web;
using System.Net;

public class UrlEncode
{
    public static void Bad(string value, HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        ctx.Response.Redirect("?param=" + WebUtility.HtmlEncode(user));
    }

    public static void Good(string value, HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        ctx.Response.Redirect("?param=" + WebUtility.UrlEncode(user));
    }
}

```
The following example shows two methods where a value is used in a SQL query. In the `Bad` method, the value is insufficiently encoded by only escaping double quotes. In the `Good` method, encoding is handled by using a SQL parameter.


```csharp
using System;
using System.Data;
using System.Data.SqlClient;
using System.Web;
using System.Net;

public class SqlEncode
{
    public static DataSet Bad(HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        using (var connection = new SqlConnection(""))
        {
            var query = "select * from Users where Name='" + user.Replace("\"", "\"\"") + "'";
            var adapter = new SqlDataAdapter(query, connection);
            var result = new DataSet();
            adapter.Fill(result);
            return result;
        }
    }

    public static DataSet Good(HttpContext ctx)
    {
        var user = WebUtility.UrlDecode(ctx.Request.QueryString["user"]);
        using (var connection = new SqlConnection(""))
        {
            var query = "select * from Users where Name=@name";
            var adapter = new SqlDataAdapter(query, connection);
            var parameter = new SqlParameter("name", user);
            adapter.SelectCommand.Parameters.Add(parameter);
            var result = new DataSet();
            adapter.Fill(result);
            return result;
        }
    }
}

```

## References
* MSDN: [HttpUtility Class](https://msdn.microsoft.com/en-us/library/system.web.httputility(v=vs.110).aspx), [HttpServerUtility Class](https://msdn.microsoft.com/en-us/library/system.web.httpserverutility(v=vs.110).aspx), [WebUtility Class](https://msdn.microsoft.com/en-us/library/system.net.webutility(v=vs.110).aspx), [How To: Protect From SQL Injection in ASP.NET](https://msdn.microsoft.com/en-us/library/ff648339.aspx).
* Common Weakness Enumeration: [CWE-838](https://cwe.mitre.org/data/definitions/838.html).
