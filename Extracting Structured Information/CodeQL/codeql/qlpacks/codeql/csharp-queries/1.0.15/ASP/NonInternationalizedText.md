# Text has not been internationalized
Some websites are internationalized, meaning they are implemented in a way that supports translation and adaption into multiple natural languages. For such a website, it is an error to embed English text directly into the page.


## Recommendation
Consider whether the text needs to be translated for each language the website supports. If so, then the text should be replaced by a code expression that retrieves the appropriate translation of the text.


## Example
In the following example, the English word "Amount" is included directly in the page.


```asp
<%@ Page Language="C#" %>

<html>
<body>
<p>Amount: <%= Amount %></p>
</body>
</html>

```
The revised example makes use of a hypothetical "I18n" library that is used to retrieve the translation of a given string of text into whichever language is currently in effect. The correct internationalization library to use will depend on the website, and might not be called exactly "I18n".


```asp
<%@ Page Language="C#" %>

<html>
<body>
<p><%= I18n.getMessage("Amount") %>: <%= Amount %></p>
</body>
</html>

```

## References
* MSDN: [Introduction to ASP.NET and Web Forms](https://msdn.microsoft.com/en-us/library/ms973868.aspx), [ASP.NET Page Syntax](https://msdn.microsoft.com/en-us/library/fy30at8h(v=vs.100).aspx).
* W3C: [Internationalization](http://www.w3.org/standards/webdesign/i18n), [Localization vs. Internationalization](http://www.w3.org/International/questions/qa-i18n).
