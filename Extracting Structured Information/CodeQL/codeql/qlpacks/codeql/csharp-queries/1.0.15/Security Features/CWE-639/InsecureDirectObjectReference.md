# Insecure Direct Object Reference
Resources like comments or user profiles can be accessed and modified through an action method. To target a certain resource, the action method accepts an ID parameter pointing to that specific resource. If the methods do not check that the current user is authorized to access the specified resource, an attacker can access a resource by guessing or otherwise determining the linked ID parameter.


## Recommendation
Ensure that the current user is authorized to access the resource of the provided ID.


## Example
In the following example, in the "BAD" case, there is no authorization check, so any user can edit any comment for which they guess or determine the ID parameter. The "GOOD" case includes a check that the current user matches the author of the comment, preventing unauthorized access.


```csharp
    // BAD - Any user can access this method.
    protected void btn1_Click(object sender, EventArgs e) {
        string commentId = Request.QueryString["Id"];
        Comment comment = getCommentById(commentId);
        comment.Body = inputCommentBody.Text;
    }

    // GOOD - The user ID is verified.
    protected void btn2_Click(object sender, EventArgs e) {
        string commentId = Request.QueryString["Id"];
        Comment comment = getCommentById(commentId);
        if (comment.AuthorName == User.Identity.Name){
            comment.Body = inputCommentBody.Text;
        }
    }
```
The following example shows a similar scenario for the ASP.NET Core framework. As above, the "BAD" case provides an example with no authorization check, and the first "GOOD" case provides an example with a check that the current user authored the specified comment. Additionally, in the second "GOOD" case, the \`Authorize\` attribute is used to restrict the method to administrators, who are expected to be able to access arbitrary resources.


```csharp
public class CommentController : Controller {
    private readonly IAuthorizationService _authorizationService;
    private readonly IDocumentRepository _commentRepository;

    public CommentController(IAuthorizationService authorizationService,
                              ICommentRepository commentRepository)
    {
        _authorizationService = authorizationService;
        _commentRepository = commentRepository;
    }

    // BAD: Any user can access this.
    public async Task<IActionResult> Edit1(int commentId, string text) {
        Comment comment = _commentRepository.Find(commentId);
        
        comment.Text = text;

        return View();
    }

    // GOOD: An authorization check is made.
    public async Task<IActionResult> Edit2(int commentId, string text) {
        Comment comment = _commentRepository.Find(commentId);
        
        var authResult = await _authorizationService.AuthorizeAsync(User, Comment, "EditPolicy");

        if (authResult.Succeeded) {
            comment.Text = text;
            return View();
        }
        else {
            return ForbidResult();
        }
    }

    // GOOD: Only users with the `admin` role can access this method.
    [Authorize(Roles="admin")]
    public async Task<IActionResult> Edit3(int commentId, string text) {
        Comment comment = _commentRepository.Find(commentId);
        
        comment.Text = text;

        return View();
    }
}
```

## References
* OWASP: [Insecure Direct Object Refrences](https://wiki.owasp.org/index.php/Top_10_2013-A4-Insecure_Direct_Object_References).
* OWASP: [Testing for Insecure Direct Object References](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/05-Authorization_Testing/04-Testing_for_Insecure_Direct_Object_References).
* Microsoft Learn: [Resource-based authorization in ASP.NET Core](https://learn.microsoft.com/en-us/aspnet/core/security/authorization/resourcebased?view=aspnetcore-7.0).
* Common Weakness Enumeration: [CWE-639](https://cwe.mitre.org/data/definitions/639.html).
