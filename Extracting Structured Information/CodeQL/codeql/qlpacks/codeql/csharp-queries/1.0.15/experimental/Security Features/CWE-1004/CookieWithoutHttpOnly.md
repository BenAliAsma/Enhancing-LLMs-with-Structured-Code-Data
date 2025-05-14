# 'HttpOnly' attribute is not set to true
Cookies without `HttpOnly` flag are accessible to JavaScript running in the same origin. In case of Cross-Site Scripting (XSS) vulnerability the cookie can be stolen by malicious script.


## Recommendation
Protect sensitive cookies, such as related to authentication, by setting `HttpOnly` to `true` to make them not accessible to JavaScript. In ASP.NET case it is also possible to set the attribute via `<httpCookies>` element of `web.config` with the attribute `httpOnlyCookies="true"`.


## Example
In the example below `Microsoft.AspNetCore.Http.CookieOptions.HttpOnly` is set to `true`.


```csharp
class MyController : Controller
{
    void Login()
    {
        var cookieOptions = new Microsoft.AspNetCore.Http.CookieOptions() { HttpOnly = true };
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
In the example below `System.Web.HttpCookie.HttpOnly` is set to `true`.


```csharp
class MyController : Controller
{
    void Login()
    {
        var cookie = new System.Web.HttpCookie("cookieName") { HttpOnly = true };
    }
}
```

## References
* [CookieOptions.HttpOnly Property,](https://docs.microsoft.com/en-us/dotnet/api/microsoft.aspnetcore.http.cookieoptions.httponly)
* [Set-Cookie](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie) Header,
* [HttpCookie.HttpOnly Property,](https://msdn.microsoft.com/en-us/library/system.web.httpcookie.httponly(v=vs.110).aspx)
* [httpCookies Element,](https://msdn.microsoft.com/library/ms228262%28v=vs.100%29.aspx)
* Common Weakness Enumeration: [CWE-1004](https://cwe.mitre.org/data/definitions/1004.html).
