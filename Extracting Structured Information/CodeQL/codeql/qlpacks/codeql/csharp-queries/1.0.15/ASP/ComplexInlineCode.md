# Complex inline code
ASP.NET allows arbitrary amounts of code to be embedded into a page. If that code is lengthy, then the overall page can be harder to read. The code itself does not fully benefit from IDE features, and the HTML content of the page is harder to read due to being obstructed by the code.


## Recommendation
Consider updating the page to use the Code-Behind Page Model (see [ASP.NET Web Page Code Model for details](https://msdn.microsoft.com/en-us/library/015103yb.aspx)). In this model the markup and programming code are stored in separate files for easier maintenance.


## Example
This example uses a large amount of code in the middle of a page.


```asp
<%@ Page Language="C#" %>

<html>
<body>
<%
  if (builder == null)
    builder = ec.GetTemporaryLocal (type);

  if (builder.LocalType.IsByRef) {
    //
    // if is_address, than this is just the address anyways,
    // so we just return this.
    //
    ec.Emit (Response, OpCodes.Ldloc, builder);
  } else {
    ec.Emit (Response, OpCodes.Ldloca, builder);
  }
%>
</body>
</html>

```
In the following example, the code is stored in a code-behind file. This separation of the HTML content from the VB.Net content makes the intention of the ASP page clearer and also simplifies reuse of the code.


```asp
<%@ Page Language="C#" %>

<html>
<body>
<%= emitLoad() %>
</body>
</html>

```

## References
* MSDN: [Introduction to ASP.NET and Web Forms](https://msdn.microsoft.com/en-us/library/ms973868.aspx), [ASP.NET Page Syntax](https://msdn.microsoft.com/en-us/library/fy30at8h(v=vs.100).aspx), [ASP.NET Web Page Code Model](https://msdn.microsoft.com/en-us/library/015103yb.aspx).
