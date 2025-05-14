# Block code with a single Response.Write()
An inline code block containing a single `Response.Write()` can be written more clearly using an inline expression.

ASP.NET provides general-purpose *inline code*, using the syntax "`<%...%>`". The inline code can emit content into the resulting HTML page by calling `Response.Write()`.

In many cases, the inline code is only one line long, and does nothing more than issue a single call to `Response.Write()`. For such cases, the call to `Response.Write()` can be longer than the code to compute what will be embedded. This makes it harder to understand the intent of the code.


## Recommendation
ASP.NET also provides *inline expressions*, using the syntax "`<%=...>`". An inline expression does not need to call `Response.Write()`. The equals sign (=) is a concise way to tell ASP.NET to call `Response.Write()`.


## Example
This example shows a page where an inline code block writes content using `Response.Write()`.


```asp
<%@ Page Language="C#" %>

<html>
<body>
<p>2 + 3 = <%Response.Write(2 + 3)%></p>
</body>
</html>

```
In the following example, the code block is replaced with an inline expression, and is thus more concise and direct.


```asp
<%@ Page Language="C#" %>

<html>
<body>
<p>2 + 3 = <%=2 + 3%></p>
</body>
</html>

```

## References
* Microsoft: [Introduction to ASP.NET inline expressions in the .NET Framework](https://support.microsoft.com/en-us/help/976112/introduction-to-asp-net-inline-expressions-in-the--net-framework).
* MSDN: [Introduction to ASP.NET and Web Forms](https://msdn.microsoft.com/en-us/library/ms973868.aspx), [ASP.NET Page Syntax](https://msdn.microsoft.com/en-us/library/fy30at8h(v=vs.100).aspx).
