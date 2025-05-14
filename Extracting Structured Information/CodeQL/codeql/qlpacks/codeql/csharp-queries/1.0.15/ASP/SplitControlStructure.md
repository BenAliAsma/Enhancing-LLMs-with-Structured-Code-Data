# Split control structure
ASP.NET is very flexible about the code that can be embedded into a page. It is not necessary that the individual code render blocks are valid syntax on their own. Rather, it is only necessary that all the code render blocks can compile as a group, once they are all emitted into a class that the ASP.NET infrastructure generates to implement the page.

This flexibility is sometimes used to implement conditional portions of an HTML page. One code render block will have the beginning of an `If...Then` statement, and another will have the `End If`, with arbitrary ASP content in between. This conditional code can make the source code of the page hard to read and maintain, especially if the embedded ASP content has further conditional code within it.


## Recommendation
Consider moving the entire block of conditional code into a code-behind file. If the embedded ASP content is more than a few lines long, consider converting the ASP content into a user control. See [ASP.NET Web Page Code Model](https://msdn.microsoft.com/en-us/library/015103yb.aspx) and [ASP.NET User Controls](https://msdn.microsoft.com/en-us/library/y6wb1a0e.aspx).


## Example
This example uses a split control structure, which can make the page harder to read.


```asp
<%@ Page Language="VB" %>

<html>
<body>
<% If ShouldWarn() Then %>
<p>WARNING: <%=warning()%></p>
<% End If %>
</body>
</html>

```
The following example moves the code to a code-behind file. By separating the HTML content from the VB.Net content, the overall body of code should be easier to read and maintain.


```asp
<%@ Page Language="VB" %>

<html>
<body>
<%=maybeEmitWarning()%>
</body>
</html>

```

## References
* MSDN: [Introduction to ASP.NET and Web Forms](https://msdn.microsoft.com/en-us/library/ms973868.aspx), [ASP.NET Page Syntax](https://msdn.microsoft.com/en-us/library/fy30at8h(v=vs.100).aspx), [ASP.NET User Controls](https://msdn.microsoft.com/en-us/library/y6wb1a0e.aspx), [ASP.NET Web Page Code Model](https://msdn.microsoft.com/en-us/library/015103yb.aspx).
