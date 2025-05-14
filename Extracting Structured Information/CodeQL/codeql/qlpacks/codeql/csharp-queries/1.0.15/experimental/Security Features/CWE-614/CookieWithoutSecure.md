# 'Secure' attribute is not set to true
Sensitive data that is transmitted using HTTP is vulnerable to being read by a third party. By default, cookies are sent via HTTP, not HTTPS.


## Recommendation
In ASP.NET case when using cookies ensure that HTTPS is used by setting the property `Microsoft.AspNetCore.Http.CookieOptions.Secure` to `true`.

In ASP.NET Core case when using cookies, ensure that HTTPS is used, either via the `<forms>` attribute above, or the `<httpCookies>` element, with the attribute `requireSSL="true"`. It is also possible to require cookies to use HTTPS programmatically, by setting the property `System.Web.HttpCookie.Secure` to `true`.


## Example
In the example below `Microsoft.AspNetCore.Http.CookieOptions.Secure` is set to `true` programmatically.


```csharp
class MyController : Controller
{
    void Login()
    {
        var cookieOptions = new Microsoft.AspNetCore.Http.CookieOptions() { Secure = true };
        Response.Cookies.Append("auth", "secret", cookieOptions);
    }
}
```
In the following example `CookiePolicyOptions` are set programmatically to configure defaults.


```csharp
public class Startup
{
    // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        app.UseCookiePolicy(new CookiePolicyOptions()
        {
            Secure = Microsoft.AspNetCore.Http.CookieSecurePolicy.Always,
            HttpOnly = Microsoft.AspNetCore.CookiePolicy.HttpOnlyPolicy.Always
        });
    }
}
```
In the example below `System.Web.HttpCookie.Secure` is set to `true` programmatically.


```csharp
class MyController : Controller
{
    void Login()
    {
        var cookie = new System.Web.HttpCookie("cookieName") { Secure = true };
    }
}
```

## References
* [CookieOptions.Secure Property,](https://docs.microsoft.com/en-us/dotnet/api/microsoft.aspnetcore.http.cookieoptions.secure)
* [Set-Cookie](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie) Header,
* [FormsAuthentication.RequireSSL Property,](https://msdn.microsoft.com/en-us/library/system.web.security.formsauthentication.requiressl(v=vs.110).aspx)
* [forms Element for authentication,](https://msdn.microsoft.com/en-us/library/1d3t3c61(v=vs.100).aspx)
* [httpCookies Element,](https://msdn.microsoft.com/library/ms228262%28v=vs.100%29.aspx)
* Common Weakness Enumeration: [CWE-319](https://cwe.mitre.org/data/definitions/319.html).
* Common Weakness Enumeration: [CWE-614](https://cwe.mitre.org/data/definitions/614.html).
