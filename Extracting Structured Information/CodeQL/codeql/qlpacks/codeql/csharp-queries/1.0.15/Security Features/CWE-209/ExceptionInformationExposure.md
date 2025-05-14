# Information exposure through an exception
Software developers often add stack traces to error messages, as a debugging aid. Whenever that error message occurs for an end user, the developer can use the stack trace to help identify how to fix the problem. In particular, stack traces can tell the developer more about the sequence of events that led to a failure, as opposed to merely the final state of the software when the error occurred.

Unfortunately, the same information can be useful to an attacker. The sequence of class names in a stack trace can reveal the structure of the application as well as any internal components it relies on. Furthermore, the error message at the top of a stack trace can include information such as server-side file names and SQL code that the application relies on, allowing an attacker to fine-tune a subsequent injection attack.


## Recommendation
Send the user a more generic error message that reveals less information. Either suppress the stack trace entirely, or log it only on the server.


## Example
In the following example, an exception is handled in two different ways. In the first version, labeled BAD, the exception is sent back to the remote user by calling `ToString()`, and writing it to the response. As such, the user is able to see a detailed stack trace, which may contain sensitive information. In the second version, the error message is logged only on the server. That way, the developers can still access and use the error log, but remote users will not see the information.


```csharp
using System;
using System.Web;

public class StackTraceHandler : IHttpHandler
{

    public void ProcessRequest(HttpContext ctx)
    {
        try
        {
            doSomeWork();
        }
        catch (Exception ex)
        {
            // BAD: printing a stack trace back to the response
            ctx.Response.Write(ex.ToString());
            return;
        }

        try
        {
            doSomeWork();
        }
        catch (Exception ex)
        {
            // GOOD: log the stack trace, and send back a non-revealing response
            log("Exception occurred", ex);
            ctx.Response.Write("Exception occurred");
            return;
        }
    }
}

```
