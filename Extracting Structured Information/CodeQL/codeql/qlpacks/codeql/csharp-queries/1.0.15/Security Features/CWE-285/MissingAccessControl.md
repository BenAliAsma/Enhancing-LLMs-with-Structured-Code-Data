# Missing function level access control
Sensitive actions, such as editing or deleting content, or accessing admin pages, should have authorization checks to ensure that they cannot be used by malicious actors.


## Recommendation
Ensure that proper authorization checks are made for sensitive actions. For WebForms applications, the `authorization` tag in `Web.config` XML files can be used to implement access control. The `System.Web.UI.Page.User` property can also be used to verify a user's role. For MVC applications, the `Authorize` attribute can be used to require authorization on specific action methods.


## Example
In the following WebForms example, the case marked BAD has no authorization checks whereas the case marked GOOD uses `User.IsInRole` to check for the user's role.


```csharp
class ProfilePage : System.Web.UI.Page {
    // BAD: No authorization is used
    protected void btn1_Edit_Click(object sender, EventArgs e) {
        ...
    }

    // GOOD: `User.IsInRole` checks the current user's role.
    protected void btn2_Delete_Click(object sender, EventArgs e) {
        if (!User.IsInRole("admin")) {
            return;
        }
        ...
    }
} 
```
The following `Web.config` file uses the `authorization` tag to deny access to anonymous users, in a `location` tag to have that configuration apply to a specific path.


```none
<?xml version="1.0"?>

<configuration xmlns:xdt="http://schemas.microsoft.com/XML-Document-Transform">
  <location path="User/Profile">
    <system.web>
      <authorization>
        <deny users="?" />
      </authorization>
    </system.web>
  </location>
</configuration>
```
In the following MVC example, the case marked BAD has no authorization checks whereas the case marked GOOD uses the `Authorize` attribute.


```csharp
public class ProfileController : Controller {

    // BAD: No authorization is used.
    public ActionResult Edit(int id) {
        ...
    }

    // GOOD: The `Authorize` attribute is used.
    [Authorize]
    public ActionResult Delete(int id) {
        ...
    }
}
```

## References
* `Page.User` Property - [Microsoft Learn](https://learn.microsoft.com/en-us/dotnet/api/system.web.ui.page.user?view=netframework-4.8.1#system-web-ui-page-user).
* Control authorization permissions in an ASP.NET application - [Microsoft Learn](https://learn.microsoft.com/en-us/troubleshoot/developer/webapps/aspnet/www-authentication-authorization/authorization-permissions).
* Simple authorization in ASP.NET Core - [Microsoft Learn](https://learn.microsoft.com/en-us/aspnet/core/security/authorization/simple?view=aspnetcore-7.0).
* Common Weakness Enumeration: [CWE-285](https://cwe.mitre.org/data/definitions/285.html).
* Common Weakness Enumeration: [CWE-284](https://cwe.mitre.org/data/definitions/284.html).
* Common Weakness Enumeration: [CWE-862](https://cwe.mitre.org/data/definitions/862.html).
